import random
import uuid
from datetime import datetime

import numpy as np
import torch as th


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M_%S"
    timestamp = datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:3]
    return f"{timestamp}_{random_uuid}"


def set_random_seed(seed: int) -> None:
    """Set random seed to both numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)