"""Reference no-op splitter for plugin authors. Production learned
splitters live in downstream packages.
"""

from .splitter import NoopSplitter

__all__ = ["NoopSplitter"]
