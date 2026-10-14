"""Create a VQA directory from unused synthetic samples with interior images.

Use stable (question_id, path_id) identities, not rewritten message contents.
Export the original VQA question/answer and sample without changing training data.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthesis.post_process.assemble_rewritten_sharegpt_dataset import iter_json_array

FULL = ROOT / 'data/sharegpt_dataset_refined_v3_rewritten_full_24k_complete/trajectories_sharegpt.json'
MERGED = ROOT / 'data/searchweave_fqva_livevqa_merged_16004_reference'
VQA = ROOT / 'runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1'
GRAPH = ROOT / 'runs/0712_multi_seed_visual_test_8192_6'
OUT = ROOT / 'data/searchweave_v2_candidates_temp'
INDEX_AUDIT = {}


def normalized(text):
    return ' '.join(re.sub(r'<image>', '', str(text or '')).split())


def index(path):
    result = {}
    duplicates = []
    total = 0
    for i, r in enumerate(iter_json_array(path), 1):
        total = i
        qid, pid = r.get('question_id'), r.get('path_id')
        key = (str(qid), str(pid)) if qid and pid else ('record_id', str(r['id']))
        human = next((c.get('value', '') for c in r.get('conversations', []) if c.get('from') == 'human'), '')
        item = {'record_id': r['id'], 'question_id': qid, 'sample_id': r.get('sample_id'),
                       'path_id': pid, 'source_index': i - 1, 'training_question': normalized(human)}
        if key in result:
            duplicates.append({'question_id': qid, 'path_id': pid,
                               'first_source_index': result[key]['source_index'],
                               'duplicate_source_index': i-1,
                               'same_record_id': result[key]['record_id'] == r['id'],
                               'same_question_text': result[key]['training_question'] == item['training_question']})
        else:
            result[key] = item
        if i % 5000 == 0:
            print(f'{path.parent.name}: indexed {i}', flush=True)
    INDEX_AUDIT[str(path)] = {'raw_records': total, 'unique_identities': len(result), 'duplicates': duplicates}
    print(f'{path.parent.name}: {total} rows, {len(result)} unique identities', flush=True)
    return result


def write_jsonl(path, records):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    summary = json.loads((MERGED / '.metadata/summary.json').read_text())
    synthetic_path = Path(summary['input_datasets'][0]['path']) / 'trajectories_sharegpt.json'
    full, merged, synthetic = index(FULL), index(MERGED / 'trajectories_sharegpt.json'), index(synthetic_path)
    assert INDEX_AUDIT[str(synthetic_path)]['raw_records'] == 13520
    assert set(synthetic) <= set(merged), 'Synthetic source no longer matches merged dataset'
    non_subset = set(synthetic) - set(full)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'identity_audit.json').write_text(json.dumps(INDEX_AUDIT, ensure_ascii=False, indent=2)+'\n')
    write_jsonl(OUT / 'synthetic_not_in_full.jsonl', (synthetic[k] for k in sorted(non_subset)))
    assert not ((set(merged) - set(synthetic)) & set(full)), 'External benchmark IDs collide with synthetic identities'
    difference = {k: r for k, r in full.items() if k not in synthetic}
    assert len(full) == len(set(full) & set(synthetic)) + len(difference)
    print(f'Difference: {len(full)} unique full - {len(set(full)&set(synthetic))} common = {len(difference)}', flush=True)
    questions = {}
    with (VQA / 'questions.jsonl').open() as f:
        for line in f:
            q = json.loads(line)
            k = (str(q['question_id']), str(q['path_id']))
            if k in difference:
                assert k not in questions, k
                assert q['sample_id'] == difference[k]['sample_id'], k
                questions[k] = q
    sample_to_key = {q['sample_id']: k for k, q in questions.items()}
    assert len(sample_to_key) == len(questions)
    found_samples, selected_questions, selected_ids, rejected = set(), [], [], []
    distribution, all_distribution = Counter(), Counter()
    manifests, mismatches = [], []
    samples_tmp = OUT / 'samples.jsonl.tmp'
    with samples_tmp.open('w') as dst, (VQA / 'samples.jsonl').open() as src:
        for source_index, line in enumerate(src):
            sample = json.loads(line)
            k = sample_to_key.get(sample.get('sample_id'))
            if k is None:
                continue
            assert k not in found_samples, k
            found_samples.add(k)
            path = sample['path']
            assert path['path_id'] == k[1], k
            ids, types = path['node_ids'], path['node_types']
            assert len(ids) == len(types) and all(t in ('image', 'text') for t in types), k
            assert all(n.startswith(t + '_') for n, t in zip(ids, types)), k
            interior = [i for i, kind in enumerate(types) if kind == 'image' and 0 < i < len(types) - 1]
            all_distribution[' -> '.join(types)] += 1
            m = dict(difference[k], original_sample_index=source_index, node_types=types,
                     interior_image_positions=interior, hop_count=len(types)-1)
            if not interior:
                rejected.append({**m, 'reason': 'no_strictly_interior_image_node'})
                continue
            q = questions[k]
            original_text = normalized(q.get('question'))
            same_question = original_text == difference[k]['training_question']
            m.update(original_question=original_text, training_question_matches_original=same_question)
            if not same_question:
                mismatches.append(m)
            # Original records remain intact: no rewriting of questions or paths.
            dst.write(line if line.endswith('\n') else line + '\n')
            selected_questions.append(q)
            selected_ids.append(k)
            manifests.append(m)
            distribution[' -> '.join(types)] += 1
            if len(manifests) % 500 == 0:
                print(f'Selected {len(manifests)} interior-image samples', flush=True)
    missing_questions = set(difference) - set(questions)
    missing_samples = set(questions) - found_samples
    missing = [{**difference[k], 'reason': 'question_not_found_in_original_vqa'} for k in sorted(missing_questions)]
    missing += [{**difference[k], 'reason': 'sample_not_found_in_original_vqa'} for k in sorted(missing_samples)]
    write_jsonl(OUT / 'unmatched_records.jsonl', missing)
    assert not missing, f'{len(missing)} unmatched original records; audit saved'
    assert len(manifests) + len(rejected) == len(difference)
    assert set(selected_ids).isdisjoint(synthetic) and set(selected_ids) <= set(full)
    assert len({q['sample_id'] for q in selected_questions}) == len(selected_questions)
    samples_tmp.replace(OUT / 'samples.jsonl')
    write_jsonl(OUT / 'questions.jsonl', selected_questions)
    write_jsonl(OUT / 'selection_manifest.jsonl', manifests)
    write_jsonl(OUT / 'difference_manifest.jsonl', difference.values())
    write_jsonl(OUT / 'excluded_no_interior_image.jsonl', rejected)
    write_jsonl(OUT / 'question_text_mismatches.jsonl', mismatches)
    (OUT / 'question_ids.txt').write_text('\n'.join(q['question_id'] for q in selected_questions) + '\n')
    overlap = {}
    for name, p in [('existing_annotation_900', ROOT.parent/'data/my_bench_test_1/annotation_ready.jsonl'),
                    ('existing_benchmark_300', ROOT.parent/'data/benchmarks/searchweave/annotation_benchmark_300.parquet')]:
        if p.suffix == '.parquet':
            import pyarrow.parquet as pq
            known = set(pq.read_table(p, columns=['question_id']).column('question_id').to_pylist())
        else:
            known = {json.loads(line)['question_id'] for line in p.read_text().splitlines() if line.strip()}
        overlap[name] = sorted(known & {q['question_id'] for q in selected_questions})
    report = {'full_records': len(full), 'merged_records': len(merged), 'merged_synthetic_records': len(synthetic),
              'raw_full_records': INDEX_AUDIT[str(FULL)]['raw_records'],
              'raw_merged_records': INDEX_AUDIT[str(MERGED / 'trajectories_sharegpt.json')]['raw_records'],
              'raw_merged_synthetic_records': INDEX_AUDIT[str(synthetic_path)]['raw_records'],
              'merged_synthetic_not_in_full': len(non_subset),
              'difference_records': len(difference), 'selected_interior_image_records': len(manifests),
              'excluded_no_interior_image': len(rejected), 'unmatched_original_records': len(missing),
              'training_original_question_text_mismatches': len(mismatches),
              'selected_path_patterns': dict(distribution), 'difference_path_patterns': dict(all_distribution),
              'selected_hop_counts': dict(Counter(m['hop_count'] for m in manifests)),
              'existing_benchmark_overlap': overlap,
              'matching_key': ['question_id', 'path_id'],
              'condition': 'At least one image node with 0 < index < len(path.nodes)-1; boundary image nodes are allowed in addition.',
              'export_question_policy': 'Preserve original VQA question and answer; training wording differences are logged, not silently substituted.',
              'paths': {'full': str(FULL), 'merged': str(MERGED), 'synthetic_source': str(synthetic_path),
                        'original_vqa': str(VQA), 'graph': str(GRAPH), 'output_vqa': str(OUT)}}
    (OUT / 'selection_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text('Temporary VQA directory from the full synthetic dataset minus merged synthetic identities.\n'
                                  'Every selected original graph path has a strictly interior image node.\n'
                                  'Original VQA questions/answers/samples are preserved. See selection_report.json for counts, wording differences and existing-benchmark overlap.\n'
                                  'Annotation export uses synthesis.benchmarks.prepare_annotation_jsonl with gemini35flash_internal_azure, 16 workers.\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('paths','existing_benchmark_overlap','difference_path_patterns')}, ensure_ascii=False, indent=2), flush=True)
    print('Existing benchmark overlap:', {k:len(v) for k,v in overlap.items()}, flush=True)


if __name__ == '__main__':
    main()
