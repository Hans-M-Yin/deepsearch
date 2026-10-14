"""Write a new tokenized cache excluding verified image/token mismatches."""
import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
from datasets import load_from_disk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenized-path', required=True, type=Path)
    p.add_argument('--alignment-report', required=True, type=Path)
    p.add_argument('--output-path', required=True, type=Path)
    p.add_argument('--cutoff-len', required=True, type=int)
    args = p.parse_args()
    if args.output_path.exists():
        raise FileExistsError(args.output_path)
    data = load_from_disk(str(args.tokenized_path))
    train = data['train']
    lengths = pc.list_value_length(train.data.table.column('input_ids')).to_numpy(zero_copy_only=False)
    required = {int(i) for i in (lengths >= args.cutoff_len).nonzero()[0]}
    rows = {}
    for line in args.alignment_report.read_text().splitlines():
        row = json.loads(line)
        rows[row['cache_index']] = row
    if required - rows.keys():
        raise ValueError('Report does not cover every cutoff-length sample')
    removed = []
    for index, row in rows.items():
        if not 0 <= index < len(train):
            raise ValueError('Report index out of range')
        sample = train[index]
        if row.get('error'):
            raise ValueError(f'Inspection failed at {index}; resolve before filtering: {row["error"]}')
        if sample['images'] != row['images'] or len(sample['input_ids']) != row['sequence_length']:
            raise ValueError(f'Report/cache mismatch at {index}')
        expected, actual = row['expected_image_tokens'], row['actual_image_tokens']
        if row['aligned'] != (expected == actual):
            raise ValueError('Inconsistent alignment report')
        if not row['aligned']:
            removed.append(index)
    bad = set(removed)
    data['train'] = train.select([i for i in range(len(train)) if i not in bad])
    data.save_to_disk(str(args.output_path))
    summary = {'source': str(args.tokenized_path.resolve()), 'report': str(args.alignment_report.resolve()),
               'before': len(train), 'after': len(data['train']), 'removed_indices': sorted(removed),
               'scope': 'cutoff-length rows covered; not a full multimodal alignment audit'}
    (args.output_path / 'alignment_filter_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
