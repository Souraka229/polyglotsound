import os
import tempfile
from pathlib import Path

# never touch the real registry from tests
os.environ.setdefault("POLYGLOT_REGISTRY", str(Path(tempfile.mkdtemp()) / "test_registry.db"))
