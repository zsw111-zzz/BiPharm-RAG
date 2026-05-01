# -*- coding: utf-8 -*-
"""
BiPharm-RAG core library

This package contains the local runtime used by the BiPharm-RAG
evaluation scripts released with the paper code.
"""

from .bipharm_rag import BiPharmRAG
from .base import QueryParam
from .tcm_prompt import apply_tcm_prompts, restore_default_prompts

__version__ = "0.1.0"

__all__ = [
    "BiPharmRAG",
    "QueryParam",
    "apply_tcm_prompts",
    "restore_default_prompts",
]
