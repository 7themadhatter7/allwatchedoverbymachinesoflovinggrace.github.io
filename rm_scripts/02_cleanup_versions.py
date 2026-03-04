#!/usr/bin/env python3
"""
RM Script 02 — Version sprawl cleanup
Moves all but the highest version of each versioned file family
to /home/joe/sparky/archive/
Safe: moves, never deletes. Review archive/ before any permanent deletion.
"""
import re
import shutil
from pathlib import Path
from collections import defaultdict

SPARKY  = Path("/home/joe/sparky")
ARCHIVE = SPARKY / "archive"
ARCHIVE.mkdir(exist_ok=True)

# Files that are actively running — never touch these
PROTECTED = {
    "mother_english_io_v5.py",
    "bridge_server.py",
    "stamp_engine.py",
    "rm_self_improvement.py",
    "combined_server.py",
    "spark_v4.py",
}

# Discover versioned families
families = defaultdict(list)
for f in SPARKY.glob("*.py"):
    if f.name in PROTECTED:
        continue
    base = re.sub(r'[_-]v\d+[a-z]?$', '', f.stem)
    if base != f.stem:           # only files that actually have a version suffix
        families[base].append(f)

moved = 0
kept  = 0

for base, files in sorted(families.items()):
    if len(files) < 2:
        continue

    files.sort(key=lambda x: int(re.search(r'v(\d+)[a-z]?$', x.stem).group(1))
               if re.search(r'v(\d+)[a-z]?$', x.stem) else 0)
    to_archive = files[:-1]   # everything except highest version
    keep       = files[-1]

    for f in to_archive:
        dest = ARCHIVE / f.name
        if dest.exists():
            dest = ARCHIVE / (f.stem + "_dup" + f.suffix)
        shutil.move(str(f), str(dest))
        moved += 1

    kept += 1
    print(f"  {base}: kept {keep.name}, archived {len(to_archive)}")

print(f"\nMoved {moved} files to {ARCHIVE}")
print(f"Kept  {kept} highest versions in place")

# Report disk delta
import subprocess
r = subprocess.run("df -h /", shell=True, capture_output=True, text=True)
print("\nDisk usage after cleanup:")
print(r.stdout.strip().splitlines()[-1])
