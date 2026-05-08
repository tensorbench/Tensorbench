"""
Configuration loading system for TensorBench.
Supports bundled defaults + user overrides + optional remote updates.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeVar, Generic, Type

from pydantic import BaseModel, ValidationError

from .models import ScenarioRequirements, HardwareCatalogEntry, UpgradeOption

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class ConfigLoader(Generic[T]):
    """Generic loader for JSON configs with validation"""
    
    def __init__(
        self,
        model_class: Type[T],
        filename: str,
        bundled_dir: Path,
        user_dir: Optional[Path] = None
    ):
        self.model_class = model_class
        self.filename = filename
        self.bundled_path = bundled_dir / filename
        self.user_path = user_dir / filename if user_dir else None
    
    def load(self) -> dict[str, T]:
        """Load and validate config, preferring user override"""
        source_path = self._resolve_source()
        
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            
            # Validate and wrap each entry
            result = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    result[key] = self.model_class(**value)
                else:
                    # Handle list/other structures if needed
                    result[key] = value
            
            logger.info(f"Loaded config '{self.filename}' from {source_path}")
            return result
            
        except FileNotFoundError:
            logger.warning(f"Config file not found: {source_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {source_path}: {e}")
            return {}
        except ValidationError as e:
            logger.error(f"Validation error in {source_path}: {e}")
            return {}
    
    def _resolve_source(self) -> Path:
        """Determine which config file to use"""
        if self.user_path and self.user_path.exists():
            return self.user_path
        if self.bundled_path.exists():
            return self.bundled_path
        raise FileNotFoundError(f"Neither {self.user_path} nor {self.bundled_path} exists")
    
    def save_user_override(self, data: dict[str, dict]) -> bool:
        """Save user-modified config to user directory"""
        if not self.user_path:
            logger.error("User config directory not set")
            return False
        
        try:
            self.user_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert models back to dicts for JSON
            serializable = {
                k: v.model_dump() if hasattr(v, 'model_dump') else v
                for k, v in data.items()
            }
            with open(self.user_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved user override: {self.user_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save user config: {e}")
            return False


# =============================================================================
# Pre-configured loaders for TensorBench
# =============================================================================

class ConfigManager:
    """High-level config manager for the application"""
    
    def __init__(self, app_data_dir: Path):
        """
        :param app_data_dir: User-writable directory (e.g. %APPDATA%/TensorBench)
        """
        # Bundled configs are next to this module
        self._bundled_dir = Path(__file__).parent.parent.parent / "config"
        self._user_dir = app_data_dir / "config"
    
    @property
    def scenarios(self) -> dict[str, ScenarioRequirements]:
        """Load AI scenario requirements"""
        loader = ConfigLoader(
            model_class=ScenarioRequirements,
            filename="scenarios.json",
            bundled_dir=self._bundled_dir,
            user_dir=self._user_dir
        )
        return loader.load()
    
    @property
    def hardware_db(self) -> dict[str, HardwareCatalogEntry]:
        """Load hardware catalog for configurator"""
        loader = ConfigLoader(
            model_class=HardwareCatalogEntry,
            filename="hardware_db.json",
            bundled_dir=self._bundled_dir,
            user_dir=self._user_dir
        )
        return loader.load()
    
    @property
    def upgrades(self) -> dict[str, UpgradeOption]:
        """Load upgrade recommendations database"""
        loader = ConfigLoader(
            model_class=UpgradeOption,
            filename="upgrades.json",
            bundled_dir=self._bundled_dir,
            user_dir=self._user_dir
        )
        return loader.load()
    
    def refresh_from_remote(self, base_url: str) -> bool:
        """
        Optional: fetch updated configs from remote server.
        Implementation deferred to later stage.
        """
        # TODO: implement HTTP download + signature verification
        logger.debug(f"Remote refresh not implemented yet (url={base_url})")
        return False