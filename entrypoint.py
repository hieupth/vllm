#!/usr/bin/env python3
"""
Entrypoint script for vLLM serving container.
- Reads config from YAML file (CONFIG_FILE env)
- Downloads model from Hugging Face Hub
- Clears cache
- Starts vLLM server
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

# =============================================================================
# Configuration
# =============================================================================
CONFIG_FILE = os.getenv("CONFIG_FILE", "/config.yaml")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================
class ConfigError(Exception):
    """Configuration related errors."""
    pass


class DownloadError(Exception):
    """Model download related errors."""
    pass


class DependencyError(Exception):
    """Missing dependency errors."""
    pass


# =============================================================================
# Config parsing
# =============================================================================
def load_config(config_path: str) -> dict[str, Any]:
    """
    Load and validate YAML config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed config dictionary

    Raises:
        ConfigError: If config is invalid or missing required fields
    """
    logger.info(f"Loading config from: {config_path}")

    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    if not os.access(path, os.R_OK):
        raise ConfigError(f"Config file is not readable: {config_path}")

    if path.stat().st_size == 0:
        raise ConfigError(f"Config file is empty: {config_path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML format: {e}")

    if config is None:
        raise ConfigError("Config file is empty or invalid")

    # Validate required fields
    if "model" not in config:
        raise ConfigError("Missing required field: 'model'")

    if not config["model"]:
        raise ConfigError("Field 'model' cannot be empty")

    return config


# =============================================================================
# Model download
# =============================================================================
def download_model(
    model_id: str,
    local_dir: str,
    token: str | None = None,
    revision: str | None = None,
) -> str:
    """
    Download model from Hugging Face Hub.

    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-3.2-1B")
        local_dir: Local directory to download to
        token: Optional HF token for private/gated models
        revision: Optional specific revision (branch/tag/commit)

    Returns:
        Path to downloaded model

    Raises:
        DownloadError: If download fails
    """
    logger.info(f"Downloading model: {model_id}")
    logger.info(f"Target directory: {local_dir}")

    # Sanitize model name for directory
    model_dir_name = model_id.replace("/", "_")
    target_dir = Path(local_dir) / model_dir_name

    # Create model directory
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            revision=revision if revision else None,
            token=token if token else None,
            resume_download=True,
        )
        logger.info(f"Model downloaded successfully to: {downloaded_path}")
        return str(downloaded_path)
    except Exception as e:
        raise DownloadError(f"Failed to download model '{model_id}': {e}")


# =============================================================================
# Cache management
# =============================================================================
def clear_cache(model_path: str | None = None) -> None:
    """
    Clear Hugging Face cache directories.

    Args:
        model_path: Optional model path to clear local .cache folder
    """
    logger.info("Clearing Hugging Face cache...")

    # Clear main HF cache directory
    hf_cache = Path(HF_CACHE_DIR)
    if hf_cache.exists():
        try:
            cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
            logger.info(f"Cache size: {cache_size / (1024**3):.2f} GB")

            shutil.rmtree(hf_cache)
            logger.info("Hugging Face cache cleared.")
        except Exception as e:
            logger.warning(f"Could not fully clear cache directory: {e}")
    else:
        logger.info(f"No cache directory found at: {HF_CACHE_DIR}")

    # Clear .cache/huggingface folder in model directory
    # (created by local_dir download per HF docs)
    if model_path:
        local_cache = Path(model_path) / ".cache" / "huggingface"
        if local_cache.exists():
            try:
                shutil.rmtree(local_cache)
                logger.info("Cleared local .cache/huggingface folder in model directory.")
            except Exception as e:
                logger.warning(f"Could not clear local cache: {e}")


# =============================================================================
# vLLM server
# =============================================================================
def build_vllm_args(config: dict[str, Any]) -> list[str]:
    """
    Build vLLM command line arguments from config.

    Args:
        config: Parsed config dictionary

    Returns:
        List of command line arguments
    """
    args = ["vllm", "serve"]

    # Get model path from config (set after download)
    model_path = config.get("_model_path")
    if not model_path:
        raise ConfigError("Model path not set in config")

    args.append(model_path)

    # Add vLLM arguments from config
    vllm_args = config.get("vllm_args", {})
    if isinstance(vllm_args, dict):
        for key, value in vllm_args.items():
            # Convert underscores to dashes for CLI args
            cli_key = key.replace("_", "-")
            args.append(f"--{cli_key}")
            args.append(str(value))
    elif isinstance(vllm_args, str) and vllm_args:
        # Allow passing raw string args
        args.extend(vllm_args.split())

    return args


def start_vllm(config: dict[str, Any]) -> None:
    """
    Start vLLM server with the specified configuration.

    Args:
        config: Parsed config dictionary
    """
    args = build_vllm_args(config)
    logger.info(f"Starting vLLM: {' '.join(args)}")

    # Use exec to replace current process with vLLM
    os.execvp(args[0], args)


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    """Main entrypoint function."""
    logger.info("=" * 50)
    logger.info("Starting vLLM Entrypoint")
    logger.info("=" * 50)

    try:
        # Step 1: Load config
        config = load_config(CONFIG_FILE)

        model_id = config.get("model", "")
        token = config.get("token") or os.getenv("HF_TOKEN")
        revision = config.get("revision")

        logger.info(f"Model: {model_id}")
        logger.info(f"Revision: {revision or 'default'}")
        logger.info(f"Token: {'[SET]' if token else '[NOT SET]'}")

        # Step 2: Download model
        model_path = download_model(
            model_id=model_id,
            local_dir=MODEL_DIR,
            token=token,
            revision=revision,
        )

        # Store model path in config for vLLM
        config["_model_path"] = model_path

        # Step 3: Clear cache
        clear_cache(model_path)

        # Step 4: Start vLLM
        logger.info("=" * 50)
        logger.info("Initialization complete. Starting server...")
        logger.info("=" * 50)
        start_vllm(config)

        return 0

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except DownloadError as e:
        logger.error(f"Download error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
