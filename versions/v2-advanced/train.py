"""Train random weights on JSONL pairs. No pretrained model or ML framework."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from gpt import GPT, cross_entropy


def encode(row, context):
    prompt = list(('BUG:\n'+row['buggy']+'\nFIX:\n').encode('utf-8'))
    answer = list(row['fixed'].encode('utf-8')) + [256]
    seq = prompt + answer
    if len(seq)-1 > context:
        raise ValueError(f'Example needs {len(seq)-1} context tokens; increase --context.')
    x, y, mask = np.zeros(context, int), np.zeros(context, int), np.zeros(context)
    x[:len(seq)-1], y[:len(seq)-1] = seq[:-1], seq[1:]
    mask[len(prompt)-1:len(seq)-1] = 1
    return x, y, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='examples.jsonl')
    ap.add_argument('--out', default='model.npz')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--width', type=int, default=48)
    ap.add_argument('--context', type=int, default=384)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--resume', help='Checkpoint from this trainer; restores Adam state when present')
    ap.add_argument('--heads', type=int, default=4)
    args = ap.parse_args()
    if args.steps < 1 or args.batch < 1:
        ap.error('Steps and batch must be positive')
    rows = [json.loads(s) for s in Path(args.data).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows) < 4:
        ap.error('Provide at least four buggy/fixed pairs')
    rng = np.random.default_rng(123)
    model = GPT.load(args.resume) if args.resume else GPT(width=args.width, context=args.context, layers=args.layers, heads=args.heads)
    args.context = model.config['context']
    groups = sorted(set(r.get('family', r['buggy']) for r in rows))
    rng.shuffle(groups)
    held = set(groups[:max(1, len(groups)//5)])
    validation = [encode(r, args.context) for r in rows if r.get('family', r['buggy']) in held]
    training = [encode(r, args.context) for r in rows if r.get('family', r['buggy']) not in held]
    if not training:
        ap.error('Need at least two distinct example families')
    m = {k: np.zeros_like(p.data) for k,p in model.p.items()}
    v = {k: np.zeros_like(p.data) for k,p in model.p.items()}
    start = 0
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as saved:
            if 'step' in saved:
                start = int(saved['step'])
                for k in m:
                    m[k][:] = saved['adam_m.'+k]
                    v[k][:] = saved['adam_v.'+k]
        print(f'Resuming at optimizer step {start}', flush=True)
    def save(step):
        np.savez_compressed(args.out, **{k:p.data for k,p in model.p.items()},
                            config=np.array(list(model.config.values())), step=step,
                            **{'adam_m.'+k:a for k,a in m.items()},
                            **{'adam_v.'+k:a for k,a in v.items()})
    print(f'{sum(p.data.size for p in model.p.values()):,} parameters; '
          f'{len(training)} training / {len(validation)} held-out examples', flush=True)
    try:
        step = start
        for local_step in range(1, args.steps+1):
            step = start + local_step
            batch = [training[j] for j in rng.integers(len(training), size=args.batch)]
            x,y,mask = (np.stack(a) for a in zip(*batch))
            # Padding is never attended to by scored positions; trim unused suffixes.
            length = int(np.where(mask)[1].max()) + 1
            x,y,mask = x[:,:length], y[:,:length], mask[:,:length]
            model.zero_grad()
            loss = cross_entropy(model.forward(x), y, mask)
            loss.backward()
            norm = np.sqrt(sum((p.grad**2).sum() for p in model.p.values()))
            scale = min(1., 1./max(norm, 1e-12))
            for k,p in model.p.items():
                g = p.grad * scale
                m[k] = .9*m[k]+.1*g
                v[k] = .999*v[k]+.001*g*g
                warmup = min(1., local_step / min(50, args.steps))
                decay = .1 + .9*.5*(1+math.cos(math.pi*local_step/args.steps))
                p.data -= args.lr*warmup*decay*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            if local_step == 1 or step % 50 == 0 or local_step == args.steps:
                vals = []
                for vx,vy,vm in validation:
                    end = int(np.where(vm)[0].max())+1
                    vals.append(float(cross_entropy(model.forward(vx[None,:end]), vy[None,:end], vm[None,:end]).data))
                print(f'Step {step}: training loss {float(loss.data):.4f}; '
                      f'held-out loss {np.mean(vals):.4f}', flush=True)
                save(step)
    except KeyboardInterrupt:
        print('Interrupted; saving current weights.')
    save(step)
    print(f'Saved {args.out}. Launch: python app.py --model "{args.out}"')


if __name__ == '__main__':
    main()
