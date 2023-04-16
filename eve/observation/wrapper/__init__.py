from enum import Enum


class MemoryResetMode(int, Enum):
    FILL = 0
    ZERO = 1


from .totrackingcs import ToTrackingCS
from .memory import Memory
from .normalize import Normalize
from .normalizecustom import NormalizeCustom
from .normalizetrackingperepisode import NormalizeTrackingPerEpisode
from .relativetofirstrow import RelativeToFirstRow
from .relativetoinsertionpoint import RelativeToInsertionPoint
from .relativetolaststate import RelativeToLastState
from .relativetotip import RelativeToTip
from .selectivememory import SelectiveMemory
