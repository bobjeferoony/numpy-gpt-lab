"""Audit tracked/publication files for local-home paths and obvious credential formats."""
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
patterns={
    'Windows home path': re.compile(rb'(?i)[a-z]:[\\/]+Users[\\/]+'),
    'Unix home path': re.compile(rb'/(?:Users|home)/[A-Za-z0-9_.-]+/'),
    'Clipboard screenshot filename': re.compile(rb'codex-clipboard-[0-9a-f-]{20,}'),
    'GitHub token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'Private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

def main():
    result=subprocess.run(['git','ls-files','-z'],cwd=root,capture_output=True)
    if result.returncode==0 and result.stdout:
        paths=[root/p.decode('utf-8') for p in result.stdout.split(b'\0') if p]
    else:
        paths=[p for p in root.rglob('*') if p.is_file() and not any(s in {'.git','.venv','.venv-win','__pycache__'} for s in p.relative_to(root).parts)]
    failures=[]
    for path in paths:
        relative=path.relative_to(root).as_posix()
        if path.suffix in {'.zip','.png'}:
            failures.append((relative,'Opaque private archive/image is not part of this public edition')); continue
        if path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as saved:
                for key in saved.files:
                    if saved[key].dtype.kind not in 'biufc':
                        failures.append((relative,'Non-numeric checkpoint array'))
            continue
        data=path.read_bytes()
        for label,pattern in patterns.items():
            if pattern.search(data): failures.append((relative,label))
    if failures:
        for path,label in failures: print(f'{path}: {label}')
        return 1
    print(f'Public-content audit passed for {len(paths)} files. Pattern scanning is not a proof of absence of all sensitive data.')
    return 0


if __name__=='__main__':
    sys.exit(main())
