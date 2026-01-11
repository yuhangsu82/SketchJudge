"""Expose factory helpers and auto-import wrappers so they self-register."""

from .base import register_model, make_model  # re-export for convenience

# Import wrappers for side-effect registration
# from . import qwen
from . import gpt
# from . import qianfan
