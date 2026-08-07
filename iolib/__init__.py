"""I/O helpers: media probing/manipulation, Drive sync, and output assembly.

Named ``iolib`` rather than ``io`` deliberately. A top-level package named ``io``
is unimportable in practice: CPython imports the stdlib ``io`` during interpreter
startup, so ``sys.modules['io']`` is already bound before any user code runs and
``from io.media import ...`` would resolve to the stdlib module and fail.

CPU-only. Safe to import anywhere.
"""

__all__ = ["media", "drive", "assembly"]
