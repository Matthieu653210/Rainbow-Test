"""
Configuration Manager using OmegaConf.

This module handles loading and managing application configuration from a YAML file (app_conf.yaml).
The configuration supports environment variable substitution and multiple environments.

Key Features:
- Loads configuration from app_conf.yaml in current directory or parent directory
- Supports environment variable substitution in config values (e.g., {oc.env:HOME})
- Provides hierarchical configuration with baseline and environment-specific overrides
- Allows runtime configuration modifications
- Implements singleton pattern for global access

Configuration File Structure:
- 'baseline' section contains base configuration
- Additional sections provide environment-specific overrides
- Environment variables can be used in values (e.g., {oc.env:HOME}/path)

Configuration selection order:
1 - selected programmatically by 'select_config"
2 - defined in BLUEPRINT_CONFIG environment variable
3 - defined by key 'default_config'
4 - baseline only


Example Usage:
    # Get a config value
    model = global_config().get("llm.default_model")
    global_config().select_config("local")  # Switch to local config
    # Set a runtime override
    global_config().set("llm.default_model", "gpt-4")
"""

# regexp :
# get_str\(([^,]+)",\s"*([^,]+)\)
# -> get_str($1.$2)

import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

from src.utils.singleton import once

load_dotenv()

CONFIG_FILE = "app_conf.yaml"
# Ensure PWD is set correctly
os.environ["PWD"] = os.getcwd()  # Hack because PWD is sometime set to a Windows path in WSL


class OmegaConfig(BaseModel):
    """Application configuration manager using OmegaConf."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: DictConfig
    selected_config: str

    @property
    def selected(self) -> DictConfig:
        return self.root.get(self.selected_config)

    @property
    def baseline(self) -> DictConfig:
        return self.root.get("baseline")

    @once
    def singleton() -> "OmegaConfig":
        """Returns the singleton instance of Config."""

        # config_loguru()
        # Find config file
        yml_file = Path.cwd() / CONFIG_FILE
        if not yml_file.exists():
            yml_file = Path.cwd().parent / CONFIG_FILE
        assert yml_file.exists(), f"cannot find config file: '{yml_file}'"
        # logger.info(f"load {yml_file}")
        config = OmegaConf.load(yml_file)
        assert isinstance(config, DictConfig)

        # Determine which config to use
        config_name_from_env = os.environ.get("BLUEPRINT_CONFIG")
        config_name_from_yaml = config.get("default_config")
        if config_name_from_env and config_name_from_env not in config:
            logger.warning(
                f"Configuration selected by environment variable 'BLUEPRINT_CONFIG' not found: {config_name_from_env}"
            )
            config_name_from_env = None
        if config_name_from_yaml and config_name_from_yaml not in config:
            logger.warning(f"Configuration selected by key 'default_config' not found: {config_name_from_yaml}")
            config_name_from_yaml = None
        selected_config = config_name_from_env or config_name_from_yaml or "baseline"
        logger.info(f"selected config={selected_config}")
        return OmegaConfig(root=config, selected_config=selected_config)

    def select_config(self, config_name: str) -> None:
        """Select a different configuration section to override defaults."""
        if config_name not in self.root:
            logger.error(f"Configuration section '{config_name}' not found")
            raise ValueError(f"Configuration section '{config_name}' not found")
        logger.info(f"Switching to configuration section: {config_name}")
        self.selected_config = config_name

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.default_model")
            default: Default value if key not found
        Returns:
            The configuration value or default if not found
        """
        # Create merged config with runtime overrides first
        merged = OmegaConf.merge(self.baseline, self.selected)
        try:
            value = OmegaConf.select(merged, key)
            if value is None:
                if default is not None:
                    return default
                else:
                    raise ValueError(f"Configuration key '{key}' not found")
            return value
        except Exception as e:
            if default is not None:
                return default
            raise ValueError(f"Configuration key '{key}' not found") from e

    def set(self, key: str, value: Any) -> None:
        """Set a runtime configuration value using dot notation.
        Args:
            key: Configuration key in dot notation (e.g., "llm.default_model")
            value: Value to set
        """
        OmegaConf.update(self.selected, key, value, merge=True)

    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get a string configuration value."""
        value = self.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"Configuration value for '{key}' is not a string (its a {type(value)})")
        return value

    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """Get a list configuration value."""
        value = self.get(key, default)
        if not isinstance(value, ListConfig):
            raise ValueError(f"Configuration value for '{key}' is not a list (its a {type(value)})")
        return list(value)

    def get_dict(self, key: str, expected_keys: list | None = None) -> dict:
        """Get a dictionary configuration value.

        Args:
            key: Configuration key in dot notation
            expected_keys: Optional list of required keys to validate against
        Returns:
            The dictionary configuration value
        Raises:
            ValueError: If value is not a dict or if expected keys validation fails
        """
        value = self.get(key)
        if not isinstance(value, DictConfig):
            raise ValueError(f"Configuration value for '{key}' is not a dict (its a {type(value)})")
        result = dict(value)
        if expected_keys is not None:
            missing_keys = [k for k in expected_keys if k not in result]
            if missing_keys:
                raise ValueError(f"Missing required keys '{key}': {', '.join(missing_keys)}")
        return result

    def get_dir_path(self, key: str, create_if_not_exists: bool = False) -> Path:
        """Get a directory path.

        Args:
            key: Configuration key containing the path
            create_if_not_exists: If True, create directory when missing
        Returns:
            The Path object
        Raises:
            ValueError: If path doesn't exist, is not a directory, or create_if_not_exists=False
        """
        path = Path(self.get_str(key))
        if not path.exists():
            if create_if_not_exists:
                logger.warning(f"Creating missing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Directory path for '{key}' does not exist: '{path}'")
        if not path.is_dir():
            raise ValueError(f"Path for '{key}' is not a directory: '{path}'")
        return path

    def get_file_path(self, key: str) -> Path:
        """Get a file path."""
        path = Path(self.get_str(key))
        if not path.exists():
            raise ValueError(f"File path for '{key}' does not exist: '{path}'")
        return path


def global_config() -> OmegaConfig:
    """Get the global config singleton."""
    return OmegaConfig.singleton()


def config_loguru() -> None:
    """
    Configure the logger.
    """

    LOGURU_FORMAT = "<cyan>{time:HH:mm:ss}</cyan>-<level>{level: <7}</level> | <magenta>{file.name}</magenta>:<green>{line} <italic>{function}</italic></green>- <level>{message}</level>"
    # Workaround "LOGURU_FORMAT" does not seems to be taken into account
    format_str = os.environ.get("LOGURU_FORMAT") or LOGURU_FORMAT
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=format_str,
        backtrace=False,
        diagnose=True,
    )


## for quick test ##
if __name__ == "__main__":
    # Get a config value
    model = global_config().get("llm.default_model")
    print(model)
    # Set a runtime override
    global_config().set("llm.default_model", "gpt-4")
    model = global_config().get("llm.default_model")
    print(model)
    # Switch configurations
    global_config().select_config("local")
    model = global_config().get("llm.default_model")
    print(model)
    print(global_config().get_list("chains.modules"))

    global_config().set("llm.default_model", "foo")
    print(global_config().get_str("llm.default_model"))

    global_config().select_config("edc_local")
    print(global_config().get_dir_path("ecod_project.datasets_root"))

    print(global_config().root.default_config)
    print(global_config().selected.llm.default_model)
