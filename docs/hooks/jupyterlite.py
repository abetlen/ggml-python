from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def on_post_build(config, **kwargs) -> None:
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])
    contents_dir = docs_dir / "jupyterlite" / "contents"
    output_dir = site_dir / "playground" / "lite"

    if not contents_dir.exists():
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory(prefix="ggml-python-jupyterlite-") as temp_dir:
        subprocess.run(
            [
                "jupyter",
                "lite",
                "build",
                "--apps",
                "lab",
                "--contents",
                str(contents_dir),
                "--output-dir",
                str(output_dir),
                "--no-libarchive",
                "--no-sourcemaps",
                "--no-unused-shared-packages",
            ],
            cwd=temp_dir,
            check=True,
        )
