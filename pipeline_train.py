"""NumPy next-byte pretraining and supervised debugging with AdamW/early stopping."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from gpt import GPT, cross_entropy
from train import encode


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['pretrain','finetune'],required=True)
    ap.add_argument('--data',default='pipeline-data')
    ap.add_argument('--out',required=True)
    ap.add_argument('--init',help='Load weights only; start a new optimizer for this stage')
    ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--width',type=int,default=128)
    ap.add_argument('--layers',type=int,default=5)
    ap.add_argument('--context',type=int,default=384)
    ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--weight-decay',type=float,default=.01)
    ap.add_argument('--eval-every',type=int,default=50)
    ap.add_argument('--patience',type=int,default=8,help='Stop after this many evaluations without improvement')
    args=ap.parse_args()
    if min(args.steps,args.batch,args.eval_every,args.patience)<1 or args.lr<=0 or args.weight_decay<0:
        ap.error('Positive steps, batch, evaluation interval, patience and LR required; decay must be nonnegative')
    if not args.out.endswith('.npz'):
        ap.error('--out must end in .npz')
    if args.stage=='finetune' and not args.init:
        ap.error('Fine-tuning requires --init pointing to pretrained weights')
    model=GPT.load(args.init) if args.init else GPT(width=args.width,layers=args.layers,context=args.context)
    context=model.config['context']
    root=Path(args.data)
    rng=np.random.default_rng(456)
    manifest={}
    if args.stage=='pretrain':
        datasets=[]
        for name in ('python-train.bin','python-validation.bin'):
            raw=(root/name).read_bytes()
            manifest[name]=hashlib.sha256(raw).hexdigest()
            datasets.append(np.frombuffer(raw,dtype=np.uint8))
        if any(len(d)<=context for d in datasets):
            ap.error('Corpus must be longer than context')
        def sample(data, generator):
            starts=generator.integers(0,len(data)-context,size=args.batch)
            seq=np.stack([data[s:s+context+1] for s in starts]).astype(int)
            return seq[:,:-1],seq[:,1:],np.ones((args.batch,context))
        training,validation=datasets
        # Fixed validation windows make measurements comparable over time.
        vrng=np.random.default_rng(789)
        val_batches=[sample(validation,vrng) for _ in range(8)]
    else:
        datasets=[]
        families=[]
        for name in ('bugs-train.jsonl','bugs-validation.jsonl'):
            raw=(root/name).read_bytes()
            manifest[name]=hashlib.sha256(raw).hexdigest()
            rows=[json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()]
            families.append({r['family'] for r in rows})
            datasets.append([encode(r,context) for r in rows])
        if families[0]&families[1] or not all(datasets):
            ap.error('Need nonempty training/validation sets with disjoint families')
        training,validation=datasets
        def pack(batch):
            x,y,mask=(np.stack(a) for a in zip(*batch))
            end=int(np.where(mask)[1].max())+1
            return x[:,:end],y[:,:end],mask[:,:end]
        def sample(data,generator):
            return pack([data[j] for j in generator.integers(len(data),size=args.batch)])
        val_batches=[pack(validation[s:s+args.batch]) for s in range(0,len(validation),args.batch)]
    m={k:np.zeros_like(p.data) for k,p in model.p.items()}
    v={k:np.zeros_like(p.data) for k,p in model.p.items()}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    best_path=out.with_name(out.stem+'.best.npz')
    last_path=out.with_name(out.stem+'.last.npz')
    log_path=out.with_suffix('.history.jsonl')
    if any(p.exists() for p in (out,best_path,last_path,log_path)):
        ap.error('Output already exists; choose a new --out to preserve previous runs')
    metadata=dict(stage=args.stage,source_hashes=manifest,config=model.config,arguments=vars(args))
    if args.init:
        metadata['initial_checkpoint_sha256']=hashlib.sha256(Path(args.init).read_bytes()).hexdigest()
    out.with_suffix('.run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    def evaluate():
        total=count=0.
        for x,y,mask in val_batches:
            n=mask.sum()
            total+=float(cross_entropy(model.forward(x),y,mask).data)*n; count+=n
        return total/count
    def save(path,step):
        np.savez_compressed(path,**{k:p.data for k,p in model.p.items()},
                            config=np.array(list(model.config.values())),step=step,
                            **{'adam_m.'+k:a for k,a in m.items()},
                            **{'adam_v.'+k:a for k,a in v.items()})
    best=evaluate(); bad=0; step=0
    save(best_path,0)
    with log_path.open('w',encoding='utf-8') as log:
        log.write(json.dumps(dict(step=0,validation_loss=best))+'\n'); log.flush()
        print(f'Initial validation loss {best:.4f}; {sum(p.data.size for p in model.p.values()):,} parameters',flush=True)
        try:
            for step in range(1,args.steps+1):
                x,y,mask=sample(training,rng)
                model.zero_grad()
                loss=cross_entropy(model.forward(x),y,mask); loss.backward()
                norm=np.sqrt(sum((p.grad**2).sum() for p in model.p.values()))
                if not np.isfinite(float(loss.data)) or not np.isfinite(norm):
                    raise FloatingPointError('Nonfinite loss or gradient; best checkpoint is preserved')
                lr=args.lr*min(1.,step/min(100,args.steps))*(.1+.45*(1+math.cos(math.pi*step/args.steps)))
                for k,p in model.p.items():
                    g=p.grad*min(1.,1/max(norm,1e-12))
                    m[k]=.9*m[k]+.1*g; v[k]=.999*v[k]+.001*g*g
                    if p.data.ndim>=2 and k not in ('token','position'):
                        p.data*=1-lr*args.weight_decay
                    p.data-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
                if step%args.eval_every==0 or step==args.steps:
                    score=evaluate()
                    improved=score<best-1e-5
                    if improved:
                        best=score; bad=0; save(best_path,step)
                    else:
                        bad+=1
                    save(last_path,step)
                    row=dict(step=step,training_loss=float(loss.data),validation_loss=score,lr=lr,best_loss=best)
                    log.write(json.dumps(row)+'\n'); log.flush()
                    print(json.dumps(row),flush=True)
                    if bad>=args.patience:
                        print('Early stopping: validation did not improve.',flush=True); break
        except KeyboardInterrupt:
            print('Interrupted; saving last weights and restoring best.',flush=True)
        finally:
            save(last_path,step)
            GPT.load(best_path).save(out)
    print(f'Best weights: {out}; last optimizer checkpoint: {last_path}',flush=True)


if __name__=='__main__':
    main()
