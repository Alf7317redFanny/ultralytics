# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.0.0"

from ultralytics.models import YOLO, SAM, FastSAM, NAS, RTDETR
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

# Personal note: I mostly use YOLO and FastSAM, keeping others for reference
__all__ = (
    "__version__",
    "YOLO",
    "SAM",
    "FastSAM",
    "NAS",
    "RTDETR",
    "SETTINGS",
    "checks",
    "download",
)
