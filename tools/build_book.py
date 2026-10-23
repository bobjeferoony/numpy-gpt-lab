"""Combine the chapter sources into one portable Markdown book."""
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
chapters=sorted((root/'docs').glob('[0-9][0-9]-*.md'))
chapters=[p for p in chapters if not p.name.startswith('00-')]
parts=['# NumPy GPT Lab: complete technical guide', '',
       'Architecture, mathematics, reasoning, experiments, operations, history and proposed feedback design.', '',
       'Compiled from the chapter files in docs/. Source code and preserved records are the implementation authority. This book distinguishes implemented behavior from future design. This is the sanitized public edition. Source snapshots and publication boundaries are described in records/PUBLICATION.md.', '',
       '## Chapter navigation', '']
for chapter in chapters:
    title=chapter.read_text(encoding='utf-8').splitlines()[0].lstrip('# ')
    parts.append(f'- [{title}](docs/{chapter.name})')
parts.append('')
for chapter in chapters:
    parts.extend(['---', '', chapter.read_text(encoding='utf-8').replace('(../assets/','(assets/'), ''])
output='\n'.join(parts)
(root/'TECHNICAL_GUIDE.md').write_text(output,encoding='utf-8')
print(f'Combined {len(chapters)} chapters; {len(re.findall(r"\S+",output)):,} whitespace-delimited words; {len(output):,} characters.')
