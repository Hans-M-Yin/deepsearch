#!/usr/bin/env python3
"""Freeze manually reviewed, suspicion-selected cases; does not call an LLM."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'runs/0712_multi_seed_visual_test_8192_8/vqa/rl_quality_filtered_benchmark_medium_hard_augmented/train_partial_credit_v3.parquet'
# IDs/steps identify the first qualifying trajectory in the frozen audit index.
# These are review hypotheses, NOT automatic ground-truth verdicts.
CASES = [
    ('eight__q_004194', 10, False, 'full_credit_wrong_location', '旧0.3；turn7起从John Willard转到Berlin/Peck Library，未建立East Guilford。检查末跳是否被错误奖励。'),
    ('eight__q_003861', 10, False, 'full_credit_wrong_entity', '旧0.3；gold为Radio France，实际围绕UNESCO/IOC搜索40轮。检查错误建筑识别是否被当作gold。'),
    ('eight__q_000698', 10, False, 'full_credit_wrong_event', '旧0.3；实际走Mořice cemetery与Operation Bagration，gold为Strašnice与1941 Kiev。共同二战背景不能替代事件关系。'),
    ('eight__q_002085', 11, False, 'full_credit_wrong_film', '旧0.3；前段Laura Belsey/DGA/Nolan有进展，turn9转向2001 A Space Odyssey，gold末跳为Star Wars。检查是否只应部分给分。'),
    ('six__q_004569', 13, False, 'full_credit_wrong_person', '旧0.3；gold为Tiago Mendes，行动持续围绕Arda Turan及Shakhtar夺冠图。检查是否应拒绝。'),
    ('eight__q_004188', 13, False, 'full_credit_wrong_city', '旧0.3；得到417后turn4走Joplin医疗机构，gold关系是Branson/Mercy。相同区号与组织不能证明同一城市。'),
    ('six__q_006253', 14, False, 'full_credit_person_conflation', '旧0.3；turn6以后从Andy Lau演员走到FAU同名物理教师，gold末跳为Jackie Chan。检查实体混淆与有效前段。'),
    ('eight__q_001121', 14, False, 'full_credit_wrong_branch', '旧0.3；Vivien Leigh后走Streetcar/Tennessee与Dakin Williams，而gold是Gone with the Wind/Margaret Mitchell。'),
    ('six__q_021689', 14, False, 'full_credit_wrong_source', '旧0.3；gold起点Inoue Kaoru，turn3后主要研究Takasugi Shinsaku及另一张合影，后面出现Ito不能自动证明gold关系。'),
    ('eight__q_002576', 20, False, 'evidence_without_adoption', '旧0.3；turn11得到Rhea/Cronus关系，turn12继续查Rhea的埃及对应而非Cronus。检查局部事实是否真正用于目标路径。'),
    ('eight__q_002037', 10, True, 'zero_credit_image_event', '旧0；turn8/9 Warsaw Uprising候选证据，turn10查Warsaw空投。注意搜索结果可能是模型场景而非原照片，不能因最终正确自动给分。'),
    ('six__q_025719', 11, True, 'zero_credit_core_vs_detail', '旧0；反搜指向Cyrene，turn3正文为Zeus神庙而gold写Apollo，turn4已查Cyrene归罗马。检查核心地点进展与神庙混淆。'),
    ('six__q_030149', 11, True, 'zero_credit_visual_evidence', '旧0；turn1反搜Greek Evzones，turn3文件标题/说明再次支持，后续查Evzones沿革。检查视觉与文字证据漏判。'),
    ('six__q_019095', 11, True, 'zero_credit_visual_evidence', '旧0；turn1反搜Clumber Park，turn3/4教堂位置正文，turn5查相关公爵画作。检查有效识别取证。'),
    ('six__q_030980', 13, True, 'zero_credit_visual_evidence', '旧0；turn1-3定位The National及查看舞台图片，turn4/5得到Berninger教育资料。检查图像身份与泛泛人物资料的区别。'),
    ('six__q_014909', 13, True, 'zero_credit_tool_image', '旧0；turn5查看CBN书封图片，turn6查CBN频道出售。检查工具图片观察后的实际使用，勿要求judge看到被省略的像素。'),
    ('six__q_028949', 13, True, 'zero_credit_partial_and_contradiction', '旧0；turn8/9明确狮子纹章对应Alfonso VII，turn10使用；后续出现Alfonso VI/VII与Two Religions混淆，需逐跳判定。'),
    ('eight__q_004467', 15, True, 'zero_credit_text_transition', '旧0；turn7/8明确Willow Pill就读Colorado State University，turn9研究该校。检查明确中间关系是否被漏掉。'),
    ('six__q_030239', 16, True, 'zero_credit_tool_image', '旧0；turn1反搜Air India，turn2看图，turn3/4查并确认Air India Building。检查识别的证据与使用时序。'),
    ('eight__q_002445', 16, True, 'zero_credit_text_transition', '旧0；turn3正文明确Vera Miles饰演Lila Crane，turn4/5查其续作命运。检查直接文本关系及首次归因。'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=Path('/tmp/failure_progress_clean_rows.json'))
    parser.add_argument('--output', type=Path, default=ROOT / 'debug/partial_credit_audit20_manifest.json')
    args = parser.parse_args()
    import pyarrow.parquet as pq
    excluded = {r['id'] for r in json.loads((ROOT / 'debug/partial_credit_audit12_manifest.json').read_text())['samples']}
    rows = json.loads(args.index.read_text())
    gold = {}
    for row in pq.read_table(ROOT / DATASET, columns=['extra_info']).to_pylist():
        e = row['extra_info']; key = e.get('source_question_id')
        statements = (e.get('partial_credit') or {}).get('statements', [])
        if key in gold and gold[key] != statements:
            raise ValueError(f'Conflicting dataset statements: {key}')
        gold[key] = statements
    samples = []
    for key, step, correct, category, note in CASES:
        assert key not in excluded
        candidates = [r for r in rows if r['id'] == key and r['step'] == step
                      and r['correct'] == correct and (r['raw'] == 0 if correct else r['raw'] >= .29)]
        if not candidates:
            raise ValueError(f'Missing reviewed case: {key}')
        row = dict(candidates[0]); path = ROOT / row['file']; blob = path.read_bytes(); rollout = json.loads(blob)
        assert rollout['data_id'] == key and bool(rollout['is_correct']) == correct
        assert row['statements'] == gold[key], key
        components = rollout['reward_components']
        assert abs(components['partial_raw_bonus'] - row['raw']) < 1e-8
        row.update(selection_category=category, selection_reason=note,
                   review_status='suspected_issue_not_gold_verdict',
                   rollout_sha256=hashlib.sha256(blob).hexdigest(),
                   historical_reward_components=components,
                   historical_statement_verdicts_available=False)
        samples.append(row)
    assert len(samples) == len({r['id'] for r in samples}) == 20
    assert sum(r['correct'] for r in samples) == 10
    manifest = {
        'description': '20 new targeted audit cases: 10 saved correct and 10 saved incorrect. Excludes all original audit12 question IDs. Suspected over/under-credit, not a representative accuracy sample or a gold-labeled judge benchmark.',
        'expected_count': 20, 'expected_correct_count': 10,
        'source_index': str(args.index), 'source_index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(),
        'statement_dataset': DATASET,
        'selection_method': 'Manual review of actions and targeted tool evidence after screening historical scores; literal target absence was only a candidate heuristic, never a correctness test.',
        'caveats': ['Saved is_correct labels are not independently rejudged here.', 'Historical per-statement verdicts and failure reasons were not saved. A zero raw bonus does not prove a negative judge verdict.', 'Images remain omitted in the text judge; visual accuracy is not independently verified.'],
        'samples': samples,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Validated 20 unique IDs, 10 correct/10 incorrect, no audit12 overlap; wrote {args.output}')


if __name__ == '__main__':
    main()
