from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_plotting_env() -> None:
    """Ensure Matplotlib/Fontconfig can reuse a writable cache directory."""
    cache_root = Path(tempfile.gettempdir()) / "babappa_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    mpl_cache = cache_root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
