"""Structured Writer responses for offline pipeline fixtures."""


def observation(reasoning, verdict="YES"):
    return {
        "reasoning": reasoning,
        "decision": "ACCEPT"
        if verdict == "YES"
        else "REJECT"
        if verdict == "NO"
        else "UNCERTAIN",
        "local_goal_satisfied": verdict == "YES",
        "established_facts": [reasoning],
        "unresolved_questions": [],
        "decision_evidence": reasoning,
    }
