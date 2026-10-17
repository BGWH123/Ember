"""Standalone source and packaged entrypoint. Safe to import under multiprocessing spawn."""
from pathlib import Path
import sys
import os

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Development only: packaged modules include torch_judge alongside this file.
if (HERE / "ember_evaluator").is_dir() and not (HERE / "torch_judge").exists():
    sys.path.insert(0, str(HERE.parents[1]))

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
    import multiprocessing
    multiprocessing.freeze_support()
    from ember_evaluator.protocol import serve
    serve()
