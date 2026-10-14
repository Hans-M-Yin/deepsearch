"""Network-free tests of the audit runner's parallel scheduling."""
import asyncio
import unittest

from debug.rejudge_partial_credit_audit12 import score_parallel


class AuditParallelTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_twelve_start_and_results_keep_manifest_order(self):
        started = []
        ready = asyncio.Event()
        release = [asyncio.Event() for _ in range(12)]
        updates = asyncio.Queue()
        snapshots = []
        async def score(candidates, config):
            index = candidates[0]
            started.append(index)
            if len(started) == 12:
                ready.set()
            await release[index].wait()
            return [{'partial_credit': {'scored': True}}]
        def save(results):
            snapshots.append(results)
            updates.put_nowait(None)
        rows = [{'id': str(i), 'raw': 0, 'bonus': 0} for i in range(12)]
        task = asyncio.create_task(score_parallel(rows, list(range(12)), None, score, save))
        try:
            await asyncio.wait_for(ready.wait(), 2)
            for i in reversed(range(12)):
                release[i].set()
                await asyncio.wait_for(updates.get(), 2)
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(snapshots[0][0]['data_id'], '11')
        self.assertEqual([r['data_id'] for r in snapshots[-1]], [str(i) for i in range(12)])
        self.assertEqual(len(snapshots), 12)


if __name__ == '__main__':
    unittest.main()
