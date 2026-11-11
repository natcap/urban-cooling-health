#!/usr/bin/env python3
"""Relocate explicitly mapped run folders without rewriting scientific outputs.

Default is a read-only preflight. --apply moves directories on the same volume
and leaves relative compatibility symlinks. --rollback reverses this mapping.
Historical manifest bytes must not be rewritten: their checksums are evidence.
"""
import argparse
import json
import os
from pathlib import Path


def migrate(root, pairs, apply=False, rollback=False):
    root = Path(root).resolve(strict=True)
    if root.name != 'UCM_official_runs':
        raise ValueError('Expected the exact UCM_official_runs directory')
    rows = list(reversed(pairs)) if rollback else pairs
    for row in rows:
        for name in ('old', 'new'):
            rel = Path(row[name])
            if rel.is_absolute() or '..' in rel.parts or str(rel) == '.':
                raise ValueError(f'Unsafe relative path: {rel}')
        old, new = root / row['old'], root / row['new']
        if rollback:
            if not old.is_symlink() or old.resolve() != new.resolve() or not new.is_dir():
                raise ValueError(f'Cannot safely roll back {old}')
        elif old.is_symlink():
            if old.resolve() != new.resolve() or not new.is_dir():
                raise ValueError(f'Unexpected compatibility link: {old}')
        elif not old.is_dir() or new.exists() or new.is_symlink():
            # Nested moves are preflighted through the parent old path.
            parent = next((p for p in pairs if row['old'].startswith(p['new'] + '/')), None)
            original = root / row['old'].replace(parent['new'], parent['old'], 1) if parent else old
            if not original.is_dir() or new.exists() or new.is_symlink():
                raise ValueError(f'Missing source or occupied destination: {old} -> {new}')
    if not apply:
        print(f'Preflight passed: {len(rows)} mapped directories; no changes')
        return
    completed = []
    try:
        for row in rows:
            old, new = root / row['old'], root / row['new']
            if rollback:
                old.unlink()  # Only the exact verified compatibility symlink.
                new.rename(old)
            elif old.is_symlink():
                continue
            else:
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
                try:
                    old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=True)
                except Exception:
                    new.rename(old)
                    raise
            completed.append(row)
    except Exception:
        # Revert this invocation only; never delete model files.
        for row in reversed(completed):
            old, new = root / row['old'], root / row['new']
            if rollback:
                old.rename(new)
                old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=True)
            else:
                old.unlink()
                new.rename(old)
        raise
    print(f'{"Rolled back" if rollback else "Migrated"}: {len(completed)} directories')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, default=Path(__file__).parent / 'configs/run-layout-20261110.json')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    migrate(args.root, json.loads(args.mapping.read_text())['moves'], args.apply, args.rollback)
