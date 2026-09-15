"""Dataset paths and experiment definitions."""

from __future__ import annotations

from pathlib import Path


DATASET_SPECS = {
    "PL-CORE": {
        "kind": "PL",
        "training": {
            label: ("只有核", "Ag2Se_濃度比(PL)", f"{label}.txt")
            for label in [10, 12, 14, 16, 24]
        },
        "target_label": 20,
        "target": ("只有核", "Ag2Se_濃度比(PL)", "20.txt"),
    },
    "PL-SHELL": {
        "kind": "PL",
        "training": {
            label: ("有核也有殼", "CS_不同濃度比(PL)", f"{label}_cs.txt")
            for label in [1, 10, 15]
        },
        "target_label": 5,
        "target": ("有核也有殼", "CS_不同濃度比(PL)", "5_cs.txt"),
    },
    "PH-CORE": {
        "kind": "PH",
        "training": {
            label: ("只有核", "Ag2Se_pH值(6~11pH)", f"Ag2Se_pH{label}_15min.txt")
            for label in [6, 8, 9, 10, 11]
        },
        "target_label": 7,
        "target": ("只有核", "Ag2Se_pH值(6~11pH)", "Ag2Se_pH7_15min.txt"),
    },
    "PH-SHELL": {
        "kind": "PH",
        "training": {
            label: ("有核也有殼", "CS_pH值(7-11 pH)", f"cs_ph{label}.txt")
            for label in [7, 8, 10, 11]
        },
        "target_label": 9,
        "target": ("有核也有殼", "CS_pH值(7-11 pH)", "cs_ph9.txt"),
    },
}


def resolve_dataset(name: str, data_root: str | Path) -> dict:
    """Resolve one public dataset definition against a local data directory."""
    spec = DATASET_SPECS[name]
    root = Path(data_root)
    return {
        "name": name,
        "kind": spec["kind"],
        "training_files": {
            label: root.joinpath(*parts)
            for label, parts in spec["training"].items()
        },
        "target_label": spec["target_label"],
        "target_file": root.joinpath(*spec["target"]),
    }
