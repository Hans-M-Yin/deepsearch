"""Apply the user-approved 72 image-end + 66 interleave exclusions.

Preserve the original benchmark and all retained rows, including image bytes.
This is a pre-refill dataset, not the completed balanced v2 benchmark.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REVIEW = Path('opensearch_infer/analysis/searchweave_v2_t2i_review')
SOURCE = Path('../data/benchmarks/searchweave/annotation_benchmark_300.parquet')
OUT = Path('data/benchmarks/searchweave')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_table(path, table):
    if path.exists():
        assert pq.read_table(path).equals(table), f'Existing output differs: {path}'
        return
    temp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, temp, compression='zstd')
    assert pq.read_table(temp).equals(table), f'Round-trip mismatch: {temp}'
    temp.replace(path)


def main():
    old = {r['case_id']: r for r in records(REVIEW / 'review.jsonl')}
    recheck = {r['case_id']: r for r in records(REVIEW / 'interleave_recheck/cases.jsonl')}
    end = {cid for cid, r in old.items() if r['category'] == 'image-end' and r['decision'] == 'filter'}
    inter = {cid for cid, r in recheck.items() if r['evidence_group'] not in ('unverified_bypass', 'judge_uncertain')}
    assert len(end) == 72 and len(inter) == 66 and not end & inter
    excluded = end | inter
    assert len(excluded) == 138
    before = digest(SOURCE)
    table = pq.read_table(SOURCE)
    ids = table.column('data_id').to_pylist()
    assert len(ids) == len(set(ids)) == 300
    assert excluded <= set(ids)
    assert set(ids) == {r['case_id'] for r in records(REVIEW.parent / 'Qwen3-VL-32B_searchweave_category_audit/samples.jsonl')}
    mask = pc.is_in(table.column('data_id'), value_set=pa.array(sorted(excluded)))
    kept = table.filter(pc.invert(mask))
    removed = table.filter(mask)
    assert kept.num_rows == 162 and removed.num_rows == 138
    assert Counter(kept.column('category').to_pylist()) == {'image-start': 100, 'image-end': 28, 'image-interleave': 34}
    assert Counter(removed.column('category').to_pylist()) == {'image-end': 72, 'image-interleave': 66}
    assert kept.schema == table.schema
    OUT.mkdir(parents=True, exist_ok=True)
    kept_path = OUT / 'annotation_benchmark_v2_filtered_162.parquet'
    removed_path = OUT / 'annotation_benchmark_v2_excluded_138.parquet'
    write_table(kept_path, kept)
    write_table(removed_path, removed)
    assert digest(SOURCE) == before, 'Original source changed during export'
    for path, expected in [(kept_path, kept), (removed_path, removed)]:
        assert pq.read_table(path).equals(expected), path
    removed_records = []
    for cid in ids:
        if cid not in excluded:
            continue
        r = old[cid]
        row = {'case_id': cid, 'category': r['category'], 'reason_code': r['reason_code'],
               'reason': r['review_note'], 'question': r['question'], 'reference_answer': r['reference_answer']}
        if cid in inter:
            rr = recheck[cid]
            row.update(reason_code=rr['evidence_group'], reason=rr['detailed_recheck'])
        removed_records.append(row)
    with (OUT / 'excluded_138.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(removed_records[0]))
        w.writeheader()
        w.writerows(removed_records)
    statuses = []
    for cid in kept.column('data_id').to_pylist():
        if cid not in old:
            status = 'image_start_unchanged'
        elif cid in recheck:
            status = 'needs_review'
        else:
            status = 'needs_review' if old[cid]['decision'] == 'hold' else 'retained_candidate'
        statuses.append({'case_id': cid, 'review_status': status})
    assert Counter(r['review_status'] for r in statuses) == {'image_start_unchanged': 100, 'retained_candidate': 30, 'needs_review': 32}
    (OUT / 'retained_162_review_status.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in statuses))
    manifest = {'source': str(SOURCE.resolve()), 'source_sha256': before,
                'filtered_dataset': str(kept_path.resolve()), 'filtered_sha256': digest(kept_path),
                'excluded_archive': str(removed_path.resolve()), 'excluded_sha256': digest(removed_path),
                'original_total': 300, 'removed_total': 138, 'remaining_total': 162,
                'removed_by_category': dict(Counter(removed.column('category').to_pylist())),
                'remaining_by_category': dict(Counter(kept.column('category').to_pylist())),
                'excluded_ids': sorted(excluded), 'retained_ids': kept.column('data_id').to_pylist(),
                'remaining_review_status': dict(Counter(r['review_status'] for r in statuses)),
                'basis': 'User authorized applying 72 image-end + 66 interleave exclusions after recheck.',
                'stage': 'Filtered, not refilled; the 32 unresolved rows remain included.',
                'validation': 'Parquet round-trip equals filtered source rows, including image bytes and original order; source SHA-256 unchanged.'}
    (OUT / 'filter_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    report = ('已应用用户确认的138条排除：image-end 72，image-interleave 66。\n'
              '过滤后数据：data/benchmarks/searchweave/annotation_benchmark_v2_filtered_162.parquet\n'
              '剩余：image-start 100，image-end 28，image-interleave 34。\n'
              '32条待核样本仍保留。本阶段尚未补齐每类100条。\n'
              '排除样本存档：annotation_benchmark_v2_excluded_138.parquet；原因见excluded_138.csv。\n'
              '历史review.jsonl/filtered_candidates.jsonl是早期建议，最终已应用ID以filter_manifest.json为准。\n'
              '原始300条数据保留；保留样本内容、图片字节、schema和原始顺序未改变。\n')
    (OUT / 'README.txt').write_text(report)
    (REVIEW / 'APPLIED_FILTER.md').write_text(report)
    print(json.dumps({k: manifest[k] for k in ('filtered_dataset', 'removed_total', 'remaining_total', 'remaining_by_category', 'remaining_review_status', 'validation')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
