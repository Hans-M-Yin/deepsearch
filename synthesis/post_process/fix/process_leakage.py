"""Conservative checks for explicit editing-process language in generated prose."""

import re
import unicodedata

IMAGE_LABEL_PATTERN = r"\bIMAGE\s*(?:\[\s*\d+\s*\]|#?\s*\d+)\b|\bIMAGE\s*\[\s*\d+\s*\]"
IMAGE_LABEL = re.compile(IMAGE_LABEL_PATTERN, re.I)
IMAGE_LABEL_INSTRUCTION = """
IMAGE 0 / IMAGE 1 and similar numbered attachment labels are request-internal
labels, not conversation content. Never copy them into reasoning. Say 'the
original image', 'the opened candidate', or identify a previously viewed image
by its visible source/title. Do not guess the referent of an ambiguous label.
Keep actual tool resource IDs (such as image_abc or img_1) unchanged.
"""


def check_image_labels(text):
    match = IMAGE_LABEL.search(unicodedata.normalize("NFKC", text))
    if match:
        raise ValueError(
            "Internal image-label leakage: "
            + repr(match.group(0))
            + ". Use a natural, unambiguous image reference instead."
        )


PATTERNS = (
    IMAGE_LABEL_PATTERN,
    r"\bdesignated_image_id\b",
    r"\b(?:references_to_check|required_reference_keys|local_visual_target|uninspected_image_ids)\b",
    r"\b(?:shortened|abbreviated|truncated|condensed|compressed)\s+(?:record|history|trajectory|context)\b",
    r"\b(?:preserved|retained)\s+(?:(?:next|original|downstream|existing)\s+)*(?:steps?|actions?|tool\s+calls?)\b",
    r"\b(?:original|rewritten|augmented|edited|preserved|patched)\s+(?:trajectory|trajectories|reasoning\s+trace)\b",
    r"\b(?:trajectory|data|dataset)\s+(?:augmentation|rewriting|editing|repair)\b",
    r"\b(?:SFT|synthetic)\s+(?:training\s+)?(?:data|dataset|sample|trajectory)\b",
    r"\b(?:splice|splicing)\s+(?:back\s+)?(?:into|to)\s+(?:the\s+)?(?:original|trajectory)\b",
    r"\b(?:original_message_\d+|editable_message_ids|required_transition_id)\b",
    r"(?:保留|保留下来|沿用)的?(?:原有|原始|后续|下一个|下一)?(?:步骤|工具调用|动作)",
    r"(?:原始|原有|改写后|增强后|修改后|修补后)的?(?:推理)?轨迹",
    r"(?:轨迹|训练数据|SFT数据)的?(?:改写|增强|编辑|修复)",
)
LEAKAGE = re.compile("|".join(PATTERNS), re.I)


def check_process_leakage(text):
    normalized = unicodedata.normalize("NFKC", text)
    match = LEAKAGE.search(normalized)
    if match:
        raise ValueError(
            "Editing-process leakage: "
            + repr(match.group(0))
            + ". Explain the next action using task evidence, not editing/preservation instructions."
        )
