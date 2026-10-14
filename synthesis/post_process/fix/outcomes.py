"""Record-level export and resume policy, independent of window status."""


def disposition(audit):
    windows = audit.get("windows", [])
    kinds = {audit.get("failure_kind")} | {w.get("failure_kind") for w in windows}
    if audit.get("discard") or kinds & {"review_rejected", "global_logic_error"}:
        return "quality_rejected"
    if audit.get("status") == "quarantined" or any(
        w.get("status") == "quarantined" for w in windows
    ):
        return "retry_pending"
    if audit.get("status") == "rewritten":
        return "augmented"
    return "keep_original"


def should_process(saved, resume=False, retry_quarantined=False):
    if saved is None:
        return True
    outcome = disposition(saved["audit"])
    # Quality failures are terminal for this input/code/config manifest.
    return outcome == "retry_pending" and (resume or retry_quarantined)


def exportable(audit):
    return disposition(audit) in {"augmented", "keep_original"}
