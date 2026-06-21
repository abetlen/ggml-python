---
title: Playground
---

# Playground

This JupyterLite notebook runs entirely in your browser and installs the bundled Pyodide wheel from the local playground package index.

<iframe id="ggml-python-playground" title="ggml-python JupyterLite playground" width="100%" height="820" style="border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px;"></iframe>

Open the <a id="ggml-python-playground-link" href="#">JupyterLite workspace</a> in a full page if the embedded view is too small.

<script>
  const ggmlPythonPlayground = "lite/lab/index.html?path=ggml-python.ipynb";
  document.getElementById("ggml-python-playground").src = ggmlPythonPlayground;
  document.getElementById("ggml-python-playground-link").href = ggmlPythonPlayground;
</script>
