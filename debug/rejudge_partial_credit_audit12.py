#!/usr/bin/env python3
"""Rejudge a frozen audit manifest (20 new cases by default); never modify training data."""
import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


async def score_parallel(samples, selected, config, score, on_completed):
    """Run trajectories concurrently; publish snapshots in manifest order."""
    completed = {}

    async def one(index, row, candidate):
        result = (await score([candidate], config))[0]
        result['data_id'] = row['id']
        result['historical_raw_bonus'] = row['raw']
        result['historical_applied_bonus'] = row['bonus']
        result['audit_selection'] = {k: row[k] for k in (
            'selection_category', 'selection_reason', 'review_status',
            'rollout_sha256', 'historical_reward_components',
            'historical_statement_verdicts_available') if k in row}
        # Selection notes and old scores are report-only: never sent to the judge.
        completed[index] = result
        # No await during snapshot/write: callbacks cannot interleave here.
        on_completed([completed[i] for i in sorted(completed)])

    tasks = [asyncio.create_task(one(i, row, candidate))
             for i, (row, candidate) in enumerate(zip(samples, selected))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'debug/partial_credit_audit20_manifest.json')
    parser.add_argument('--rollout-root', type=Path, default=ROOT,
                        help='Root containing RL/rollouts_saved (for copied repositories).')
    parser.add_argument('--output', type=Path, default=Path('/tmp/partial_credit_audit20_rejudge.json'))
    parser.add_argument('--concurrency', type=int, default=20, help='Maximum concurrent judge requests (default: 20).')
    parser.add_argument('--dry-run', action='store_true', help='Validate all inputs without importing/calling the judge.')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    manifest = json.loads(args.manifest.read_text())
    selected = []
    for row in manifest['samples']:
        path = args.rollout_root / row['file']
        blob = path.read_bytes()
        if row.get('rollout_sha256') and hashlib.sha256(blob).hexdigest() != row['rollout_sha256']:
            raise ValueError(f'Rollout changed since selection: {path}')
        rollout = json.loads(blob)
        if rollout.get('data_id') != row['id']:
            raise ValueError(f'Data ID mismatch: {path}')
        if bool(rollout['is_correct']) != row['correct']:
            raise ValueError(f'Outcome changed: {path}')
        messages = rollout.get('messages', [])
        question = next((str(m.get('content') or '') for m in messages
                         if m.get('role') == 'user' and '<tool_response>' not in str(m.get('content'))), '')
        if not question or not row['statements']:
            raise ValueError(f'Missing question/statements: {path}')
        task = {'question': question, 'extra_info': {'partial_credit': {'statements': row['statements']}}}
        selected.append((path, rollout, task))
        print(f"[audit-input] id={row['id']} correct={row['correct']} old_raw={row['raw']:.3f} statements={len(row['statements'])}", flush=True)
    expected = manifest.get('expected_count', 12)
    if len(selected) != expected or len({str(x[0]) for x in selected}) != expected:
        raise ValueError(f'Expected exactly {expected} distinct trajectories')
    if len({r['id'] for r in manifest['samples']}) != expected:
        raise ValueError('Audit requires distinct question IDs')
    if sum(r['correct'] for r in manifest['samples']) != manifest.get('expected_correct_count', 6):
        raise ValueError('Unexpected correct/incorrect balance')
    if args.dry_run:
        print(f'Validated {expected} inputs; no judge requests sent.')
        return
    if args.output.exists():
        raise FileExistsError(f'Report exists; choose another --output: {args.output}')

    from score_saved_partial_credit import _score
    from vision_deepresearch_async_workflow import partial_credit_reward as pc

    config = replace(pc.PartialCreditConfig.from_env(), mode='shadow',
                     judge_concurrency=args.concurrency, include_judge_reasons=True)
    pc.validate_partial_credit_judge_model(config)
    prompt = pc._JUDGE_SYSTEM_PROMPT + pc._JUDGE_REASON_INSTRUCTION
    report = {'description': manifest['description'], 'config': asdict(config),
              'judge_protocol': 'quoted_memory_core_relation_input_image_v1',
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'judge_system_prompt': prompt, 'results': [],
              'audit_manifest': manifest,
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def run():
        started = time.perf_counter()
        def save_completed(results):
            report['results'] = results
            # Save after each trajectory, so an interrupted run retains completed judgments.
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(f'[audit-progress] completed={len(results)}/{len(selected)}', flush=True)

        print(f'[audit] trajectory_tasks={len(selected)} judge_concurrency={config.judge_concurrency}; chunks within each trajectory remain sequential', flush=True)
        await score_parallel(manifest['samples'], selected, config, _score, save_completed)
        report['summary'] = {
            'selected': len(selected),
            'wall_elapsed_s': time.perf_counter() - started,
            'scored': sum(bool(r['partial_credit'].get('scored')) for r in report['results']),
            'note': 'Compare raw_bonus, not shadow bonus=0. Historical per-statement verdicts were not saved; this is not an unbiased precision estimate.',
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    asyncio.run(run())
    print(f'[audit] report={args.output}')


if __name__ == '__main__':
    main()
