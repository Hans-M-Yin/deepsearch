# Final ShareGPT quality filter

Run after augmentation has finished exporting its JSON. This independent gate
covers every exported record, including unchanged originals. It reuses the fix2
multimodal quality judge without exposing reference answers or prior judgments.

```bash
python -m dotenv -f synthesis/.env run -- \
  python -m synthesis.post_process.fix2.filter_sharegpt \
  --input synthesis/.ignore/i2i_fix_pilot_20260918_gemini_v34_test/trajectories_sharegpt.json \
  --workdir synthesis/.ignore/i2i_fix_v34_final_quality_v1 \
  --model-alias gemini38flash_internal_azure \
  --workers 16 --max-tokens 2400 --resume
```

Add `--limit 128` for a pilot; remove it and resume in the same workdir for all
records. Worker count may change. Input contents, image root, model, max tokens,
and reviewer code are fingerprinted; changing those requires a new workdir.
Never replace the input while the filter is running. A workdir lock prevents
concurrent runs. Successful keep/filter checkpoints are reused; errors retry on
the next invocation. Each checkpoint records elapsed seconds and the decision.

Images are bound to their actual message indices, sent without local resizing,
and deduplicated by the judge input builder. Search thumbnails that were never
viewed are not added. Text-only questions are supported. Full text is sent with
no local truncation; context/API errors remain retry_pending, not quality rejects.
Image paths must exist; missing images and malformed records are execution errors.
For relative image paths specify `--image-root` (default: input JSON directory).

Outputs:

- `trajectories_sharegpt.json`: kept records only, in input order. Conversation
  text is unchanged; relative image paths become absolute so the new output
  directory does not break image references.
- `reviews.json`: all decisions with input indices and record IDs.
- `rejected.json`: filter decisions/reasons, not copies of the original records.
- `retry_pending.json`: execution errors excluded from the training output.
- `records/{index}.json`: atomic per-record checkpoints.
- `summary.json`: selected/unprocessed/export counts and dispositions.

The source dataset is not overwritten. Final outputs are each atomically replaced
after the selected batch finishes; interrupted runs retain their checkpoints,
while existing exported files still represent the previous completed batch.
This is an evidence-quality gate, not a guarantee of factual correctness.
