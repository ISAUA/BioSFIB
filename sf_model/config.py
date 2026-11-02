"""Configuration loading for the SSR release."""
from pathlib import Path
import yaml


def load_config(path):
    def read(current, stack):
        current = Path(current).resolve()
        if current in stack:
            raise ValueError(f"Circular base_config: {current}")
        with current.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        base = config.pop("base_config", None)
        if base:
            parent = read(current.parent / base, stack + (current,))
            config = merge(parent, config)
        return config

    config = read(path, ())
    model = config.get("model", {})
    if model.get("architecture", "ssr") not in ("ssr", "ssr_tri"):
        raise ValueError("Only the SSR architecture is included in this release")
    if model.get("output_fusion_mode", "ssr") != "ssr":
        raise ValueError("The formal embedding must be SSR z_out")
    if model.get("sfib_spectral_operator", "legacy_gft") != "legacy_gft":
        raise ValueError("This release uses the physical GFT operator")
    return config


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result
