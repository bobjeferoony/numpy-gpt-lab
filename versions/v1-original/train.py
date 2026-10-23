"""Train random weights on JSONL pairs. No pretrained model or ML framework."""
import argparse
import json
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
    ap.add_argument('--context', type=int, default=192)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=.001)
    args = ap.parse_args()
    if args.steps < 1 or args.batch < 1:
        ap.error('Steps and batch must be positive')
    rows = [json.loads(s) for s in Path(args.data).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows) < 4:
        ap.error('Provide at least four buggy/fixed pairs')
    rng = np.random.default_rng(123)
    rng.shuffle(rows)
    split = max(1, len(rows)//5)
    validation = [encode(r, args.context) for r in rows[:split]]
    training = [encode(r, args.context) for r in rows[split:]]
    model = GPT(width=args.width, context=args.context, layers=args.layers)
    m = {k: np.zeros_like(p.data) for k,p in model.p.items()}
    v = {k: np.zeros_like(p.data) for k,p in model.p.items()}
    print(f'{sum(p.data.size for p in model.p.values()):,} parameters; '
          f'{len(training)} training / {len(validation)} held-out examples', flush=True)
    try:
        for step in range(1, args.steps+1):
            batch = [training[j] for j in rng.integers(len(training), size=args.batch)]
            x,y,mask = (np.stack(a) for a in zip(*batch))
            model.zero_grad()
            loss = cross_entropy(model.forward(x), y, mask)
            loss.backward()
            norm = np.sqrt(sum((p.grad**2).sum() for p in model.p.values()))
            scale = min(1., 1./max(norm, 1e-12))
            for k,p in model.p.items():
                g = p.grad * scale
                m[k] = .9*m[k]+.1*g
                v[k] = .999*v[k]+.001*g*g
                p.data -= args.lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            if step == 1 or step % 50 == 0 or step == args.steps:
                vals = []
                for vx,vy,vm in validation:
                    vals.append(float(cross_entropy(model.forward(vx[None]), vy[None], vm[None]).data))
                print(f'Step {step}: training loss {float(loss.data):.4f}; '
                      f'held-out loss {np.mean(vals):.4f}', flush=True)
                model.save(args.out)
    except KeyboardInterrupt:
        print('Interrupted; saving current weights.')
    model.save(args.out)
    print(f'Saved {args.out}. Launch: python app.py --model "{args.out}"')


if __name__ == '__main__':
    main()
