# VQA Generation Progress

This file tracks the implementation status of the `synthesis/vqa` package.

## Goal

Build an automatic pipeline that turns an existing multimodal graph into
multi-hop question samples:

`graph -> path sampling -> evidence building -> obfuscation -> question writing -> polishing -> verification`

## Design Decisions Locked In

- The first version uses one unified pipeline rather than separate pipelines for
  different trajectory types.
- Trajectories are sampled randomly from the graph with minimal validity
  constraints.
- Trajectory type is recorded as metadata and analyzed later, rather than being
  hard-constrained during first-pass sampling.
- We will still support adding benchmark-oriented targeted trajectories later.

## Module Status

| Module | File | Responsibility | Status | Notes |
|---|---|---|---|---|
| Package export | `__init__.py` | Public exports for the VQA package | Done | Initial exports added |
| Schemas | `schemas.py` | Shared dataclasses for paths, evidence, drafts, verification, samples | Done | May expand once writer/verifier become concrete |
| Graph view | `graph_view.py` | Read-only graph index over `JsonlGraphStore` | Done | Basic node/edge lookup is ready |
| Path sampler | `path_sampler.py` | Random path sampling and trajectory labeling | In Progress | `SamplerConfiguration`, `generate()`, exact dedup, edge-penalized sampling, and run stats are implemented |
| Evidence builder | `evidence_builder.py` | Build oracle/writer/verifier evidence bundles from sampled paths | Scaffolded | Currently only basic text/image field extraction |
| Obfuscation | `obfuscation.py` | Pre/post anti-leakage rewriting hooks | Scaffolded | Only simple string masking exists now |
| Question writer | `question_writer.py` | LLM-backed draft + polish interface | Scaffolded | Uses placeholder backend for now |
| Verifier | `verifier.py` | Structural / semantic / shortcut checks | Scaffolded | Only minimal non-empty checks currently |
| Pipeline | `pipeline.py` | Orchestrate end-to-end generation | Scaffolded | Can already run with placeholder components |

## Current Progress Summary

### Completed

- Created the `synthesis/vqa` package.
- Defined stable intermediate schemas.
- Added a read-only graph adapter.
- Added a first-pass random path sampler.
- Added a first-pass evidence builder.
- Added orchestration glue for the full pipeline.
- Added sampler configuration and sampler run statistics.
- Added exact-signature dedup and edge-usage penalty during random walk.

### In Progress

- Tightening path validity constraints beyond basic short/dead-end/cycle filtering.
- Deciding exact dedup strategy for paths and samples.
- Designing writer prompts and verifier prompts.

### Not Started

- Real LLM-backed question drafting.
- Real question polishing.
- Multi-stage verification:
  - no-search filtering
  - text-only filtering
  - answer uniqueness
  - leakage detection
- Dataset-level dedup and diversity reporting.
- Export runner / CLI.

## Next Recommended Implementation Order

1. Strengthen `evidence_builder.py`
2. Implement prompt-backed `question_writer.py`
3. Implement real `verifier.py`
4. Add dataset export + reporting utilities
5. Revisit richer sampler validity scoring if needed

## Open Questions

- Exact minimum validity constraints for a sampled path
- Whether to reject paths with zero modality switch in the first release
- How aggressive pre-obfuscation should be before question drafting
- Which verifier to use first:
  - cheap LLM verifier
  - agent-based verifier
  - both
