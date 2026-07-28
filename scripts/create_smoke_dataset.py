from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def create_smoke_dataset(root: Path) -> None:
    root = root.expanduser().resolve()
    images = {
        "left": Image.new("RGB", (64, 48), "#325c80"),
        "right": Image.new("RGB", (64, 48), "#39725d"),
        "depth_fsd": Image.new("I;16", (64, 48), 1500),
        "depth_color": Image.new("RGB", (64, 48), "#8c633d"),
    }
    for folder, image in images.items():
        destination = root / folder / "frame_0001.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)

    cloud = root / "ply" / "frame_0001.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "-0.1 -0.1 1.0 255 80 80",
                "0.1 -0.1 1.0 80 255 80",
                "-0.1 0.1 1.0 80 80 255",
                "0.1 0.1 1.0 255 255 255",
                "",
            )
        ),
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a minimal five-view smoke-test dataset.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    create_smoke_dataset(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
