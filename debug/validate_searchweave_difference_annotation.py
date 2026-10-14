"""Validate the completed temporary VQA/annotation export without API calls."""
import json
from collections import Counter
from pathlib import Path

P = Path('data/searchweave_v2_candidates_temp')


def rows(name):
    return [json.loads(l) for l in (P / name).read_text().splitlines() if l.strip()]


def bilingual_fields(value, path=''):
    if isinstance(value, dict):
        if 'en' in value and 'zh' in value and value['en']:
            yield path, value
        for k, v in value.items():
            yield from bilingual_fields(v, path + '.' + k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from bilingual_fields(v, path + f'[{i}]')


def main():
    q, s, m, a = rows('questions.jsonl'), rows('samples.jsonl'), rows('selection_manifest.jsonl'), rows('annotation_ready.jsonl')
    assert len(q) == len(s) == len(m) == len(a) == 1669
    expected = {(r['question_id'], r['path_id']): r for r in q}
    assert len(expected) == 1669
    assert {(r['question_id'], r['path_id']) for r in a} == set(expected)
    assert len({r['sample_id'] for r in s}) == 1669
    assert {r['sample_id'] for r in s} == {r['sample_id'] for r in q}
    empty_zh, statuses, warnings = [], Counter(), Counter()
    for r in a:
        original = expected[(r['question_id'], r['path_id'])]
        assert r['question']['en'] == original['question']
        assert r['answer']['en'] == original['answer']
        nodes = r['search_chain']['nodes']
        assert any(n['node_type'] == 'image' and 0 < i < len(nodes)-1 for i, n in enumerate(nodes)), r['question_id']
        for path, field in bilingual_fields(r):
            if not str(field['zh']).strip():
                empty_zh.append({'question_id': r['question_id'], 'field': path})
        statuses[r['record_status']] += 1
        warnings.update(w.split(':', 1)[0] for w in r.get('warnings', []))
    report = json.loads((P / 'annotation_ready.report.json').read_text())
    latest = {r['record_key']: r for r in rows('annotation_ready.translation_cache.jsonl')}
    relevant = {f"{r['question_id']}::{r['path_id']}" for r in a}
    translation_status = Counter(latest[k]['status'] for k in relevant)
    validation = {'selected_unique_samples': 1669, 'annotation_records': len(a),
                  'all_have_interior_image': True, 'all_questions_answers_preserved': True,
                  'record_status_counts': dict(statuses), 'warning_type_counts': dict(warnings),
                  'translation_status_counts': dict(translation_status),
                  'translation_report_errors': report['translation']['errors'],
                  'empty_chinese_fields': empty_zh}
    (P / 'export_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    assert not empty_zh, 'Empty Chinese fields; see export_validation.json'
    assert translation_status == {'complete': 1669}, 'Incomplete translations; rerun export to retry cached failures'


if __name__ == '__main__':
    main()
