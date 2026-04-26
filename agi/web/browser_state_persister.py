# browser_state_persister.py
import json
import logging
from pathlib import Path
from typing import Any, Callable
from playwright.async_api import BrowserContext
from .browser_types import STATE_SNAPSHOT_FILENAME, PLAYWRIGHT_STORAGE_STATE_FILENAME

logger = logging.getLogger(__name__)


class BrowserStatePersister:
    """
    负责所有状态和数据的持久化工作。
    """
    
    def __init__(self, storage_dir: Path, restored_snapshot: dict[str, Any] | None):
        self.storage_dir = storage_dir
        self._state_snapshot_path = self.storage_dir / STATE_SNAPSHOT_FILENAME
        self._playwright_storage_state_path = self.storage_dir / PLAYWRIGHT_STORAGE_STATE_FILENAME
        self._restored_state_snapshot = restored_snapshot

    def load_persisted_snapshot(self) -> dict[str, Any] | None:
        if not self._state_snapshot_path.exists():
            return None
        try:
            data = json.loads(self._state_snapshot_path.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            logger.debug("Failed to load persisted browser state snapshot", exc_info=True)
            return None

    async def persist_playwright_storage_state(self, context: BrowserContext) -> None:
        if context is None or not hasattr(context, "storage_state"):
            return
        try:
            await context.storage_state(path=str(self._playwright_storage_state_path))
        except Exception:
            logger.debug("Failed to persist Playwright storage state", exc_info=True)

    def persist_state_snapshot(self, snapshot_func: Callable[[], dict[str, Any]]) -> None:
        snapshot = snapshot_func()
        try:
            self._state_snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
            self._restored_state_snapshot = snapshot
        except Exception:
            logger.debug("Failed to persist browser state snapshot", exc_info=True)

    def get_persistent_paths(self) -> tuple[Path, Path]:
        return self._state_snapshot_path, self._playwright_storage_state_path
