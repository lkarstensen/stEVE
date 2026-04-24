from enum import Enum


class MemoryResetMode(int, Enum):
    FILL = 0
    ZERO = 1


from .memory import Memory
from .normalize import Normalize
from .normalizecustom import NormalizeCustom
from .normalizetracking2depisode import NormalizeTracking2DEpisode
from .relativetofirstrow import RelativeToFirstRow
from .relativetolaststate import RelativeToLastState
from .selectivememory import SelectiveMemory
