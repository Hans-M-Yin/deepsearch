"""Tool adapter with private Top-K override and durable original resource metadata."""

from dataclasses import asdict
from pathlib import Path
import shutil
from threading import BoundedSemaphore
from .schema import Candidate
from .storage import digest, file_digest
from .matching import same_metadata


class ToolBackend:
    def __init__(self, root, config, cache):
        self.root, self.config, self.cache = Path(root), config, cache
        self.visual_slots = BoundedSemaphore(config.global_visual_workers)

    def _context(self, identity, metadata=None):
        from synthesis.sft.api_tools import ToolRuntimeContext
        from synthesis.sft import tools

        class Context(ToolRuntimeContext):
            def postprocess_search_output(self, tool_name, output, *, timings=None):
                compact, resources = tools.postprocess_search_output(
                    tool_name=tool_name,
                    output=output,
                    extract_url_keywords=False,
                    timings=timings,
                )
                for resource in resources:
                    self.register_url_resource(resource)
                return compact

        return Context(
            working_dir=str(self.root / identity),
            case_id=identity,
            metadata={
                **(metadata or {}),
                "i2i_private_result_limit": self.config.replay_top_k,
            },
        )

    def search(self, source, call, metadata=None, nonce="first"):
        inputs = {
            "source": source,
            "image_hash": file_digest(source),
            "call": call,
            "top_k": self.config.replay_top_k,
            "metadata": metadata or {},
            "nonce": nonce,
        }

        def compute():
            from synthesis.sft.api_tools import execute_tool_call

            identity = "fix_" + digest(inputs)[:20]
            ctx = self._context(identity, metadata)
            ref = ctx.register_image(source)
            args = {
                **call["arguments"],
                "image": ref,
                "top_k": self.config.replay_top_k,
            }
            args.pop("url", None)
            result = execute_tool_call("i2i_search", args, ctx)
            if result.output.get("ok") is False:
                raise RuntimeError(str(result.output))
            candidates = []
            for index, item in enumerate(
                result.output.get("matches", result.output.get("results", []))
            ):
                resources = [
                    asdict(ctx.resolve_resource_id(item[k]))
                    for k in ("image_id", "page_id", "source_page_id")
                    if item.get(k) and ctx.resolve_resource_id(item[k])
                ]
                candidates.append(
                    asdict(Candidate(identity + "_" + str(index), item, resources))
                )
            return {
                "candidates": candidates,
                "requested": self.config.replay_top_k,
                "received": len(candidates),
                "raw_observation": result.output,
            }

        saved = self.cache.get_or_compute("search", inputs, compute)
        return [Candidate(**c) for c in saved["candidates"]]

    def read(self, candidate):
        if candidate.image_paths:
            return candidate  # Historical images are reused without network calls.
        # Keep durable copies on the shared filesystem, not a worker's /tmp.
        image_root = Path(__file__).resolve().parents[4] / "tmp" / "i2i_fix_images"
        inputs = {
            "candidate": candidate.key,
            "resources": candidate.resources,
            "image_store": str(image_root),
            "image_store_version": 1,
        }

        def compute():
            from synthesis.sft.api_tools import execute_tool_call
            from synthesis.sft.tools import UrlResource

            ctx = self._context("read_" + digest(inputs)[:20])
            for data in candidate.resources:
                ctx.register_url_resource(UrlResource(**data))
            result = execute_tool_call(
                "read_url",
                {
                    "resource_id": candidate.public["image_id"],
                    "goal": "Inspect candidate image",
                },
                ctx,
            )
            paths = [str(Path(p).resolve()) for p in result.new_images.values()]
            if not paths:
                raise ValueError("Candidate did not produce an image")
            destination = image_root / digest(inputs)
            destination.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            for index, path in enumerate(paths):
                saved = destination / f"{index}_{Path(path).name}"
                shutil.copy2(path, saved)
                saved_paths.append(str(saved))
            return {"paths": saved_paths, "observation": result.output_text}

        saved = self.cache.get_or_compute("read", inputs, compute)
        candidate.image_paths = saved["paths"]
        candidate.read_observation = saved["observation"]
        return candidate


def recover_historical(window, replayed, dataset_root, record=None):
    """Do not attach a merely similar-title image to a historical result."""
    from .trajectory_index import resolve_path

    historical = []
    for item in window.candidates:
        candidate = Candidate("old_" + item["image_id"], dict(item), historical=True)
        read = window.reads.get(item["image_id"])
        if read:
            candidate.originally_viewed = True
            candidate.image_paths = [
                resolve_path(p, dataset_root) for p in read["paths"]
            ]
            candidate.read_observation = read["observation"]
        else:
            # Exact public metadata equality is conservative provenance alignment;
            # visual verification still decides whether the entity is correct.
            matches = [c for c in replayed if same_metadata(c.public, item)]
            if len(matches) == 1:
                recovered = matches[0]
                if record is not None:
                    from .provenance import page_alignment

                    if not page_alignment(item, recovered, record):
                        historical.append(candidate)
                        continue
                candidate.resources = recovered.resources
                # Use the actual recovered image ID; preserve historical page ID.
                candidate.public["image_id"] = recovered.public["image_id"]
                # This is a retained historical card, not a fresh search card.
                # Its text may motivate immutable downstream searches. Never
                # replace historical keywords with replay metadata (often empty).
                candidate.key = "recovered_" + candidate.key
        historical.append(candidate)
    return historical
