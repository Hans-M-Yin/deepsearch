import json
from pathlib import Path

from synthesis.sft.merge_sharegpt_datasets import merge_datasets


def _write_dataset(root: Path, name: str, *, row_id: str, image_name: str) -> Path:
    dataset = root / name
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / image_name).write_bytes(b"fake-image-" + name.encode())
    (dataset / "trajectories_sharegpt.json").write_text(
        json.dumps(
            [
                {
                    "id": row_id,
                    "question_id": row_id,
                    "conversations": [],
                    "images": [f"images/{image_name}"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (dataset / "dataset_info.json").write_text(
        json.dumps({"opensearch_vl_sft": {"file_name": "trajectories_sharegpt.json"}}),
        encoding="utf-8",
    )
    (dataset / ".metadata").mkdir()
    (dataset / ".metadata" / "summary.json").write_text("{}\n", encoding="utf-8")
    (dataset / ".metadata" / "rejected.jsonl").write_text("", encoding="utf-8")
    return dataset


def test_merge_rewrites_images_and_preserves_duplicate_ids(tmp_path: Path) -> None:
    first = _write_dataset(tmp_path, "part1", row_id="same", image_name="one.jpg")
    second = _write_dataset(tmp_path, "part2", row_id="same", image_name="two.jpg")
    output = tmp_path / "merged"

    summary = merge_datasets([first, second], output, workers=2)

    rows = json.loads((output / "trajectories_sharegpt.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["images"][0].startswith("images/dataset00_part1/")
    assert rows[1]["images"][0].startswith("images/dataset01_part2/")
    assert (output / rows[0]["images"][0]).is_file()
    assert (output / rows[1]["images"][0]).is_file()
    assert summary["duplicate_ids_preserved"] == 1
    assert summary["copied_image_files"] == 2
