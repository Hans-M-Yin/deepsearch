"""Run a bounded title-recovery experiment without changing training trajectories."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict
from .storage import Cache, read_json, write_json
from .schema import Config
from .replay_i2i import ToolBackend
from .title_recovery import search_title
from .matching import same_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", default="3,5,6,8")
    args = parser.parse_args()
    backend = ToolBackend(
        args.output_dir / "resources", Config(), Cache(args.output_dir / "cache")
    )

    def probe(index):
        audit = read_json(args.run_dir / "records" / f"{index}.json")["audit"]
        window = audit["windows"][0]
        anchor_id = window["anchor"]["anchor_image_id"]
        original = next(
            c["public"]
            for c in window["visual_candidates"]
            if c["historical"] and c["public"]["image_id"] == anchor_id
        )
        result = {"index": index, "original": original}
        try:
            candidates = search_title(backend, original["title"])
            result.update(
                candidates=[asdict(c) for c in candidates],
                exact_metadata_matches=[
                    c.key for c in candidates if same_metadata(original, c.public)
                ],
            )
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)
        write_json(args.output_dir / f"{index}.json", result)
        print(
            {
                "index": index,
                "count": len(result.get("candidates", [])),
                "matches": len(result.get("exact_metadata_matches", [])),
                "error": result.get("error"),
            },
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(probe, [int(x) for x in args.indices.split(",")]))
    write_json(args.output_dir / "summary.json", results)


if __name__ == "__main__":
    main()
