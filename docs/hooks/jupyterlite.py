from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

GGML_PYTHON_WHEEL_INDEX = "https://abetlen.github.io/ggml-python/whl/cpu/ggml-python/"
# The current Pyodide wheel is tagged for the 2026 ABI, while JupyterLite 0.7
# defaults to a 2025 ABI Pyodide runtime.
PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/"
PYODIDE_CORE_URL = (
    "https://github.com/pyodide/pyodide/releases/download/314.0.0/"
    "pyodide-core-314.0.0.tar.bz2"
)
PYODIDE_WHEEL_SUFFIX = "-py3-none-pyemscripten_2026_0_wasm32.whl"


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)


def _find_latest_pyodide_wheel_url() -> str:
    with urllib.request.urlopen(GGML_PYTHON_WHEEL_INDEX, timeout=30) as response:
        html = response.read().decode("utf-8")

    parser = _LinkParser()
    parser.feed(html)

    for href in parser.hrefs:
        if href.endswith(PYODIDE_WHEEL_SUFFIX):
            return href

    msg = f"could not find Pyodide wheel in {GGML_PYTHON_WHEEL_INDEX}"
    raise RuntimeError(msg)


def _download_pyodide_wheel(wheels_dir: Path) -> Path:
    wheel_url = _find_latest_pyodide_wheel_url()
    wheel_path = wheels_dir / wheel_url.rsplit("/", 1)[-1]
    with urllib.request.urlopen(wheel_url, timeout=30) as response:
        wheel_path.write_bytes(response.read())
    return wheel_path


def _configure_pyodide_runtime(output_dir: Path) -> None:
    config_path = output_dir / "jupyter-lite.json"
    config_data = json.loads(config_path.read_text())
    lite_settings = config_data["jupyter-config-data"].setdefault(
        "litePluginSettings", {}
    )
    kernel_settings = lite_settings.setdefault(
        "@jupyterlite/pyodide-kernel-extension:kernel", {}
    )
    kernel_settings["pyodideUrl"] = "./static/pyodide/pyodide.mjs"
    load_options = kernel_settings.setdefault("loadPyodideOptions", {})
    load_options["indexURL"] = PYODIDE_INDEX_URL
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")


def _patch_pyodide_kernel_extension(output_dir: Path) -> None:
    extension_dir = (
        output_dir
        / "extensions"
        / "@jupyterlite"
        / "pyodide-kernel-extension"
        / "static"
    )
    dynamic_import_stub = (
        '476:e=>{function t(e){return Promise.resolve().then((()=>{var t=new '
        "Error(\"Cannot find module '\"+e+\"'\");throw "
        't.code="MODULE_NOT_FOUND",t}))}t.keys=()=>[],t.resolve=t,t.id=476,'
        "e.exports=t}"
    )

    for path in extension_dir.glob("*.js"):
        text = path.read_text(errors="replace")
        patched = text.replace("{type:void 0}", '{type:"module"}')
        patched = patched.replace(
            dynamic_import_stub,
            "476:e=>{e.exports=e=>import(e)}",
        )
        patched = patched.replace(
            '["sqlite3","ipykernel","comm","pyodide_kernel","jedi","ipython"]',
            '["ipykernel","comm","pyodide_kernel","jedi","ipython"]',
        )

        if patched != text:
            path.write_text(patched)


def on_post_build(config, **kwargs) -> None:
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])
    contents_dir = docs_dir / "jupyterlite" / "contents"
    output_dir = site_dir / "playground" / "lite-2026"

    if not contents_dir.exists():
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    with tempfile.TemporaryDirectory(prefix="ggml-python-jupyterlite-") as temp_dir:
        wheels_dir = Path(temp_dir) / "wheels"
        wheels_dir.mkdir()
        pyodide_wheel = _download_pyodide_wheel(wheels_dir)

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
                "--piplite-wheels",
                str(pyodide_wheel),
                "--pyodide",
                PYODIDE_CORE_URL,
                "--no-libarchive",
                "--no-sourcemaps",
                "--no-unused-shared-packages",
            ],
            cwd=temp_dir,
            check=True,
        )

    _configure_pyodide_runtime(output_dir)
    _patch_pyodide_kernel_extension(output_dir)
