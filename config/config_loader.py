import yaml
from pathlib import Path

class Config:
    _cache = None

    @staticmethod
    def load(path: str = "config/settings.yaml"):
        if Config._cache is None:
            filepath = Path(path)
            if not filepath.exists():
                raise FileNotFoundError(f"Settings file not found: {filepath}")

            with open(filepath, "r") as f:
                Config._cache = yaml.safe_load(f)

        return Config._cache
