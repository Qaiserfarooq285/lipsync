"""CPU-only orchestration layer.

Nothing in this package may import torch, CUDA libraries, or any model runtime at
module level. Heavy work lives in ``gpu/`` and is imported lazily from inside the
stage functions so that ``core`` stays importable on a machine with no GPU.
"""

__all__ = ["config", "consent", "scriptgen", "pipeline", "queue"]
