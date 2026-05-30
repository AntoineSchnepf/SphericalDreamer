"""
Shared environment and logging setup for all pipeline scripts.

Import this module as early as possible (before other heavy imports) so that
GLOG env vars are set before any library that reads them is initialised, and
so that the 360monodepth source directory is on sys.path.

The default path is hardcoded below. Override it by setting the
MONODEPTH360_SRC environment variable.

Usage:
    import pipeline.bootstrap  # noqa: F401
"""
import os
import sys
import logging
import warnings

_MONODEPTH360_SRC_DEFAULT = "./submodules/360monodepth/code/python/src"
sys.path.append(os.environ.get("MONODEPTH360_SRC", _MONODEPTH360_SRC_DEFAULT))

os.environ["GLOG_minloglevel"] = "2"
os.environ["GLOG_logtostderr"] = "0"
os.environ["CERES_MINIMIZER_PROGRESS_TO_STDOUT"] = "0"

logging.disable(logging.CRITICAL + 1)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", FutureWarning)
