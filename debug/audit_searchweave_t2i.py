"""Assemble saved evidence for a human review of SearchWeave's t2i necessity.

No inference requests and no edits to the original benchmark. Review decisions
are maintained separately, so an evaluator label is never a manual-review claim.
"""
import json
import csv
from collections import Counter
from pathlib import Path

from analyze_searchweave_categories import structured
from searchweave_t2i_review_notes import NOTES

BASE = Path('opensearch_infer/outputs')
OUT = Path('opensearch_infer/analysis/searchweave_v2_t2i_review')
GPT = BASE / 'GPT5/searchweave_no_tools'
ABL = BASE / 'Qwen3-VL-8B-SFT13_5-RL5-step110-ablation/searchweave_no_t2i_i2i'
CACHE = Path('opensearch_infer/analysis/Qwen3-VL-32B_searchweave_category_audit')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index(path):
    records = rows(path)
    result = {r['case_id']: r for r in records}
    assert len(result) == len(records), path
    return result


def trace(d):
    calls = []
    for t in d.get('turns', []):
        call = structured(t.get('tool_call'))
        if not isinstance(call, dict) or t.get('tool_output_type') == 'max_turns_finalization':
            continue
        calls.append({'turn': t.get('turn'), 'name': call.get('name'),
                      'parameters': call.get('parameters', call.get('arguments', {})),
                      'reasoning': t.get('response_text', ''),
                      'output': t.get('tool_output', '')})
    return calls


def main():
    samples = index(CACHE / 'samples.jsonl')
    baseline = index(CACHE / 'traces.jsonl')
    gpt, ablation = index(GPT / 'llm_eval_report_details.jsonl'), index(ABL / 'llm_eval_report_details.jsonl')
    evidence = []
    for case_id, s in sorted(samples.items()):
        if s['category'] not in ('image-end', 'image-interleave'):
            continue
        g = gpt[case_id]
        assert g['category'] == s['category']
        gd = json.loads((GPT / f'{case_id}_trajectory.json').read_text())
        assert gd.get('mode') == 'no_tools', (case_id, gd.get('mode'))
        assert not trace(gd), case_id
        original = gd['original_data']
        metadata = structured(original.get('source_metadata', {}))
        record = {k: s[k] for k in ('case_id', 'category', 'question', 'reference_answer', 'hop_count', 'path_nodes', 'input_image_count', 'file')}
        record.update(gpt5=g, gpt5_file=str(GPT / f'{case_id}_trajectory.json'),
                      source_metadata=metadata, baseline_calls=baseline[case_id]['calls'],
                      baseline_answer=s['model_answer'], baseline_correct=s['correct'])
        a = ablation.get(case_id)
        if a:
            assert (a['question'], a['correct_answer'], a['category']) == (g['question'], g['correct_answer'], g['category']), case_id
        files = [ABL / f'{case_id}_trajectory.json']
        files += sorted((ABL / 'failures').glob(f'{case_id}*_failure.json'))
        p = next((p for p in files if p.exists()), None)
        ad = json.loads(p.read_text()) if p else {}
        if ad:
            assert set(ad.get('disabled_tools', [])) == {'t2i_search', 'i2i_search'}, case_id
            assert ad.get('image_download_disabled') is True, case_id
            ac = trace(ad)
            assert not any(c['name'] in ('t2i_search', 'i2i_search') for c in ac), case_id
            assert not any('读取图片如下' in str(c['output']) for c in ac), case_id
        else:
            ac = []
        record.update(ablation_judge=a, ablation_file=str(p) if p else None,
                      ablation_status=ad.get('status'), ablation_failure=ad.get('failure_reason'),
                      ablation_calls=ac, ablation_answer=ad.get('final_response_text', ''))
        record['screen_stage'] = ('gpt5_no_tools_correct' if g['acc'] == 1 else
                                  'text_ablation_correct' if a and a['acc'] == 1 else
                                  'remaining_manual_review')
        evidence.append(record)
    assert len(evidence) == len({r['case_id'] for r in evidence}) == 200
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evidence.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in evidence))
    summary = {'total': 200, 'by_category': {}, 'no_new_inference': True,
               'primary_no_tools': str(GPT), 'primary_ablation': str(ABL),
               'original_benchmark_modified': False,
               'scope': 'Necessary active text-to-image retrieval; input-image recognition alone does not establish t2i necessity.',
               'caveat': 'Saved successful bypasses support filtering; unsuccessful runs do not establish necessity.'}
    for cat in ('image-end', 'image-interleave'):
        summary['by_category'][cat] = dict(Counter(r['screen_stage'] for r in evidence if r['category'] == cat))
    (OUT / 'screening_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    export_review(evidence, samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def export_review(evidence, all_samples):
    manual_ids = {r['case_id'] for r in evidence if r['screen_stage'] == 'remaining_manual_review'}
    assert manual_ids == set(NOTES), (manual_ids - set(NOTES), set(NOTES) - manual_ids)
    review = []
    for r in evidence:
        g, a = r['gpt5'], r['ablation_judge']
        stage = r['screen_stage']
        if stage == 'gpt5_no_tools_correct':
            note = ('filter', 'gpt5_no_tools_correct', 'high',
                    '按用户指定的筛选规则：GPT-5在未调用工具的保存运行中被判答对，因此不纳入强调主动搜图的v2候选。此记录不单独区分记忆、推断、猜测或judge误判。')
            method = 'user_rule_on_saved_label'
        elif stage == 'text_ablation_correct':
            note = ('filter', 'text_ablation_correct', 'high',
                    'GPT-5 no-tools未被判对，但RL5保存的消融运行被判答对；该运行禁用了t2i/i2i及后续图片下载。初始输入图片仍保留，这支持无需主动搜图即可取得答案，不表示完全无需视觉。')
            method = 'user_rule_on_verified_ablation'
        else:
            note = NOTES[r['case_id']]
            method = 'manual_question_and_saved_trace_review'
        decision, reason, confidence, explanation = note
        anchors = []
        terms = {'q_001393': ['DXSA 7206'], 'q_013742': ['crossed out the pre-printed'],
                 'q_028672': ['hammer at the foot', "sculptor’s hammer discarded"]}.get(r['case_id'], [])
        for key, path in [('ablation_calls', r['ablation_file']), ('baseline_calls', r['file'])]:
            for c in r[key]:
                output = str(c['output'])
                for term in terms:
                    pos = output.casefold().find(term.casefold())
                    if pos >= 0:
                        anchors.append({'file': path, 'turn': c['turn'], 'tool': c['name'],
                                        'excerpt': output[max(0, pos - 120):pos + len(term) + 220]})
                        break
        item = {k: r[k] for k in ('case_id', 'category', 'question', 'reference_answer', 'hop_count', 'path_nodes', 'input_image_count', 'screen_stage')}
        item.update(decision=decision, reason_code=reason, confidence=confidence,
                    review_method=method, review_note=explanation, evidence_anchors=anchors,
                    gpt5_acc=g['acc'], gpt5_answer=g['model_answer'], gpt5_judge_reason=g['reasoning'],
                    ablation_acc=a['acc'] if a else None,
                    ablation_answer=a['model_answer'] if a else None,
                    ablation_judge_reason=a['reasoning'] if a else None,
                    ablation_tool_counts=dict(Counter(c['name'] for c in r['ablation_calls'])),
                    baseline_acc=r['baseline_correct'],
                    baseline_tool_counts=dict(Counter(c['name'] for c in r['baseline_calls'])),
                    sources={'gpt5': r['gpt5_file'], 'ablation': r['ablation_file'], 'baseline': r['file']})
        review.append(item)
    assert len(review) == 200
    assert all(r['gpt5_acc'] == 0 for r in review if r['review_method'].startswith('manual'))
    assert next(r for r in review if r['case_id'] == 'q_027772')['input_image_count'] == 0
    for name, records in [('review', review), ('filtered_candidates', [r for r in review if r['decision'] == 'filter']),
                          ('retained_candidates', [r for r in review if r['decision'] == 'keep_candidate']),
                          ('held_cases', [r for r in review if r['decision'] == 'hold'])]:
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    fields = ['case_id', 'category', 'decision', 'reason_code', 'confidence', 'review_method',
              'gpt5_acc', 'ablation_acc', 'baseline_acc', 'input_image_count', 'hop_count',
              'review_note', 'question', 'reference_answer', 'gpt5_answer', 'ablation_answer',
              'gpt5_judge_reason', 'ablation_judge_reason', 'evidence_anchors', 'sources']
    with (OUT / 'review.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in review:
            w.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], (dict, list)) else r[k] for k in fields})
    stats = {'total': 200, 'decisions': dict(Counter(r['decision'] for r in review)),
             'reason_codes': dict(Counter(r['reason_code'] for r in review)), 'categories': {},
             'rule_screened': 138, 'manual_reviewed_remaining': 62,
             'new_inference_requests': 0, 'new_web_searches': 0,
             'direct_image_inspections_in_this_review': 0,
             'ablation_missing_judgments': sum(r['ablation_acc'] is None for r in review),
             'retained_is_provisional': True, 'original_benchmark_modified': False}
    for cat in ('image-end', 'image-interleave'):
        rr = [r for r in review if r['category'] == cat]
        stats['categories'][cat] = {'total': len(rr), 'decisions': dict(Counter(r['decision'] for r in rr)),
                                    'screen_stages': dict(Counter(r['screen_stage'] for r in rr))}
    (OUT / 'review_summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    starts = sorted(r['case_id'] for r in all_samples.values() if r['category'] == 'image-start')
    assert len(starts) == 100
    (OUT / 'image_start_unchanged_ids.txt').write_text('\n'.join(starts) + '\n')
    (OUT / 'retained_candidate_ids.txt').write_text('\n'.join(r['case_id'] for r in review if r['decision'] == 'keep_candidate') + '\n')
    (OUT / 'filtered_candidate_ids.txt').write_text('\n'.join(r['case_id'] for r in review if r['decision'] == 'filter') + '\n')
    lines = ['# SearchWeave v2：主动文本搜图必要性首轮筛查', '',
             '范围：image-end、image-interleave各100条。GPT-5 no-tools作为首轮筛选模型；'
             'RL5-step110 no_t2i_i2i作为文本消融证据。使用已有Qwen3-VL-32B完整轨迹作辅助审查。', '',
             '目标是必须主动检索并观察目标图片才能取得关键线索。仅需识别已给输入图，不能证明t2i_search必要。'
             '类别名image-interleave对应讨论中的mix-interleaving。', '',
             '| 类别 | 原始 | GPT-5 no-tools过滤 | 文本消融额外过滤 | 人工额外过滤建议 | 保留候选 | 待核/改写 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for cat, s in stats['categories'].items():
        stages, decisions = s['screen_stages'], s['decisions']
        lines.append(f"| {cat} | 100 | {stages.get('gpt5_no_tools_correct', 0)} | {stages.get('text_ablation_correct', 0)} | {decisions.get('filter', 0) - stages.get('gpt5_no_tools_correct', 0) - stages.get('text_ablation_correct', 0)} | {decisions.get('keep_candidate', 0)} | {decisions.get('hold', 0)} |")
    lines += ['', '## 判定口径与局限', '',
              '- 前138条按用户规则由已有正确标签筛出；没有声称这些标签都经过逐条重新判分。',
              '- 剩余62条逐条结合问题、标注路径、已保存回答及轨迹审查，具体理由见review.csv。',
              '- 人工建议中observed_text_evidence是已保存文字直接出现关键答案；text_bypass_likely是结构性绕过判断，尚未取得成功文本实证。两者不能混同。',
              '- high/medium/low为本次筛选判断的置信度，不是统计概率或对网页事实的独立验证。',
              '- RL5轨迹逐条核对disabled_tools、image_download_disabled及实际调用。关闭t2i时read_url也禁止新增图片；原始输入图保留。',
              '- RL6同名目录未有相应消融标记，本次不作为禁用搜图的证据。',
              '- 缺失的消融判分不按答错处理；不能从失败推出视觉必要性，也不能从猜对推出定位正确。',
              '- 本轮没有新LLM请求、网页搜索或源图逐张核对。keep_candidate表示适合下一步保留/核验，不证明文本绝对不可搜到。',
              '- 原始benchmark未删除或改写，未补齐样本、未发布正式v2；本目录是可审查的筛选结果。', '',
              '## 额外发现的直接文本绕过', '',
              '- q_001393：消融turn25正文直接含DXSA 7206，最终模型却错答A Rock Opera。',
              '- q_013742：完整轨迹turn0/1文本直接给Sutherland划去、重写、签注的纠错过程，无需人物图片。',
              '- q_028672：消融turn12/24/25文字直接含锤子细节，最终也答hammer，但judge仍判错。', '',
              '## 需要剔除/修复的质量问题示例', '',
              '- q_027772：题目要求识别pictured antenna，实际输入图为0。',
              '- q_027899：题目二选一VITAME VAS/MATICKA PRAHA，金标却为Welcome to。',
              '- q_025468：某Carrefour门店价格没有门店、时间或视频身份，无法唯一锁定图片。',
              '- q_021719（待核）：问题与image-node元数据给出不同画家。',
              '- q_029632（待核）：没有指定地图版本，配色可不同。', '',
              '## 交付文件', '',
              'review.csv和review.jsonl：全部200条的决定、理由、答案与源文件路径。',
              'evidence.jsonl：问题、完整标注元数据、完整工具轨迹及判分，支持追溯。',
              'filtered_candidates.jsonl、retained_candidates.jsonl、held_cases.jsonl：按决定分组。',
              'audit_checks.ipynb：重算统计和校验分组覆盖。']
    (OUT / 'review.md').write_text('\n'.join(lines) + '\n')
    code = """import json
from pathlib import Path
from collections import Counter
p = Path.cwd()
if not (p / 'review.jsonl').exists():
    p = Path('opensearch_infer/analysis/searchweave_v2_t2i_review')
rows = [json.loads(line) for line in (p / 'review.jsonl').read_text().splitlines()]
assert len(rows) == len({r['case_id'] for r in rows}) == 200
assert Counter(r['category'] for r in rows) == {'image-end':100, 'image-interleave':100}
for decision, name in [('filter','filtered_candidates'), ('keep_candidate','retained_candidates'), ('hold','held_cases')]:
    ids = {json.loads(line)['case_id'] for line in (p / (name + '.jsonl')).read_text().splitlines()}
    assert ids == {r['case_id'] for r in rows if r['decision'] == decision}
print(Counter(r['decision'] for r in rows))
print(Counter(r['screen_stage'] for r in rows))
"""
    notebook = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# SearchWeave t2i review checks\n', 'Saved-label screening and manual evidence review; no new inference.']},
                          {'cell_type': 'code', 'metadata': {}, 'execution_count': 1,
                           'source': code.splitlines(keepends=True), 'outputs': []}],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'nbformat': 4, 'nbformat_minor': 5}
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    notebook['cells'][1]['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': buf.getvalue().splitlines(keepends=True)}]
    (OUT / 'audit_checks.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
