from typing import Type
from cheesechaser.datapool import IncrementIDDataPool


def quick_webp_pool(site_name: str, level: int = 3) -> Type[IncrementIDDataPool]:
    """Generic cheesechaser pool for any deepghs/<site>-webp-4Mpixel mirror."""
    repo_id = f"deepghs/{site_name}-webp-4Mpixel"

    class _QuickWebpDataPool(IncrementIDDataPool):
        def __init__(self, revision: str = "main"):
            """Initialize the cheesechaser data pool against the deepghs '<site>-webp-4Mpixel' mirror repo."""
            IncrementIDDataPool.__init__(
                self, data_repo_id=repo_id, data_revision=revision,
                idx_repo_id=repo_id, idx_revision=revision, base_level=level)

    return _QuickWebpDataPool
