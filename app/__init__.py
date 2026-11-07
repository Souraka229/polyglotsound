import os

# Colab / Jupyter export MPLBACKEND=module://matplotlib_inline.backend_inline, a module that does not
# exist in our isolated venv: matplotlib then refuses to import. The UI renders figures off-screen.
if os.environ.get("MPLBACKEND", "").startswith("module://") or "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
