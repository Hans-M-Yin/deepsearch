"""CPU-only sequence audit at actor RPC entry; no extra model forward."""
import hashlib
import json


def sequence_fingerprints(input_ids, attention_mask):
    ids = input_ids.detach().cpu()
    mask = attention_mask.detach().cpu().bool()
    return [hashlib.sha256(json.dumps(row[valid].tolist(), separators=(",", ":")).encode()).hexdigest()
            for row, valid in zip(ids, mask, strict=True)]


def verify_actor_input(data, stage):
    expected = data.non_tensor_batch.get("tito_sequence_fingerprint")
    if expected is None or len(expected) != len(data.batch["input_ids"]):
        raise ValueError(f"[TITO_ACTOR_ALIGNMENT_ERROR] stage={stage} missing sequence fingerprints")
    actual = sequence_fingerprints(data.batch["input_ids"], data.batch["attention_mask"])
    for row, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left != right:
            raise ValueError(f"[TITO_ACTOR_ALIGNMENT_ERROR] stage={stage} row={row} sequence changed after assembly")
    responses = data.batch["responses"]
    if responses.shape[-1] and not data.batch["input_ids"][:, -responses.shape[-1]:].equal(responses):
        raise ValueError(f"[TITO_ACTOR_ALIGNMENT_ERROR] stage={stage} response/input suffix mismatch")
