#!/usr/bin/env python
"""Build a reusable crystal-graph cache for a CIFData dataset.

The cache stores only graph tensors and material IDs. Targets are still read
from each dataset's id_prop.csv at run time, so one cache can be shared by a
single-task and a multitask view of the same MP structures.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cgcnn.data import CIFData


def build_shard(job):
    root, start, end, shard_index, output_dir = job
    dataset = CIFData(root)
    items = []
    for idx in range(start, end):
        structures, _, cif_id = dataset[idx]
        items.append((structures, cif_id))
    output_dir = Path(output_dir)
    output_path = output_dir / f'shard_{shard_index:05d}.pt'
    torch.save(items, output_path)
    return shard_index, start, end, output_path.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    root = args.root_dir.resolve()
    dataset = CIFData(str(root))
    output_dir = root / '.cgcnn_cache'
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f'{output_dir} already exists; use --overwrite')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shard_size = args.shard_size
    jobs = []
    for shard_index, start in enumerate(range(0, len(dataset), shard_size)):
        end = min(start + shard_size, len(dataset))
        jobs.append((str(root), start, end, shard_index, str(output_dir)))

    started = time.time()
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_shard, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                'shard {:05d}: rows [{}:{}), {:.1f} MB'.format(
                    result[0], result[1], result[2], result[3] / 1024**2,
                ),
                flush=True,
            )

    completed.sort()
    manifest = {
        'root_dir': str(root),
        'dataset_size': len(dataset),
        'shard_size': shard_size,
        'shards': len(completed),
        'cache_format': 'list[(structures, cif_id)]',
        'elapsed_seconds': round(time.time() - started, 3),
    }
    (output_dir / 'index.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
