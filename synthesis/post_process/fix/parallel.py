"""Bounded record scheduling; do not enqueue an entire training dataset."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


def completed_records(function, indices, workers):
    iterator = iter(indices)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}

        def submit():
            try:
                index = next(iterator)
            except StopIteration:
                return
            pending[executor.submit(function, index)] = index

        for _ in range(workers):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                yield index, future.result()
                submit()
