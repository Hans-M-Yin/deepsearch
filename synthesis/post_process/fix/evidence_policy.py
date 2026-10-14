"""Shared target-relevance rules for visual checking, writing and review."""

TARGET_EVIDENCE = """
Judge evidence relative to the LOCAL objective, not a generic image-difference
checklist. Distinguish person identity, a particular object instance, object
category, and photograph correspondence. An exact-photo search can be a method
of identifying someone; failing to find the same photo does not reject identity.
For each decisive observation explain: what differs/matches in the original and
candidate, why it matters to the local target, and what conclusion it supports.
When the objective is identity, actively compare identity-discriminating features
BEFORE settling for 'different photograph'. If the visible evidence establishes
different people or object instances, explicitly conclude that they are different;
do not automatically weaken a supported identity exclusion to uncertainty.
For people, compare corresponding visible features such as nose bridge/tip shape,
ear contours, eye/eyebrow spacing, jaw/chin structure and facial proportions.
These are optional comparison directions, not a required checklist or claims
about these particular images. Give the actual paired observations and explain
why viewpoint, expression or lighting does not adequately account for them.
One decisive, well-resolved distinction can suffice; do not require a fixed count.
State separately what is established about the photograph and about identity.
For a person, prioritize clearly visible facial structure and feature shapes or
relative positions when informative. Consider viewpoint, expression, age,
lighting, occlusion and resolution before treating a difference as diagnostic.
Clothing, background, pose, hairstyle and facial hair can change for the SAME
person; they alone do not establish different identity. Do not invent fine facial
or body-shape differences to satisfy this instruction. If the face cannot be
compared reliably, say identity remains unverified rather than claiming mismatch.
For objects, use target-relevant geometry, component arrangement, distinctive
markings or other discriminative structure; shared category or color alone does
not establish the same instance. Clothing details ARE relevant if the garment
itself or a clothing-defined scene is the requested target.
Garments, accessories and background alignment may corroborate same-photograph
correspondence; explicitly distinguish this from recognizing a named person.
Identity names require the source/caption evidence, not buttons or visual likeness.
Do not start from a supplied label and search for arbitrary differences to justify
it. If evidence only establishes a different photograph, retain that narrower
conclusion. Prefer one target-diagnostic observation over many incidental details.
Do not reverse-engineer stronger evidence for a preselected 'different person'
verdict. If a diagnostic feature is not resolved in both images, do not invent it.
"""
