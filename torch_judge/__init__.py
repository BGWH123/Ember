"""Pyre — PyTorch 刷题判定引擎。在 Jupyter Notebook 中使用。

Usage:
    from torch_judge import check, status

    # 查看所有题目进度
    status()

    # 实现完函数后，运行判定
    check("relu")
"""

from torch_judge._version import __version__
from torch_judge.engine import check, hint
from torch_judge.progress import status, reset_progress

__all__ = ["__version__", "check", "hint", "status", "reset_progress"]
