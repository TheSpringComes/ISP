"""YAML 配置加载器."""

import yaml
import os
from typing import Dict, Any


class Config:
    """简单的层级配置, 从 YAML 文件加载."""

    def __init__(self, cfg_dict: Dict = None):
        self._cfg = cfg_dict or {}

    def __getattr__(self, key):
        if key.startswith('_'):
            return super().__getattribute__(key)
        val = self._cfg.get(key)
        if isinstance(val, dict):
            return Config(val)
        return val

    def __getitem__(self, key):
        return self._cfg[key]

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def to_dict(self) -> Dict:
        return self._cfg

    @classmethod
    def from_file(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def merge(self, other: Dict):
        """递归合并另一个 dict."""
        def _merge(base, new):
            for k, v in new.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    _merge(base[k], v)
                else:
                    base[k] = v
        _merge(self._cfg, other)
        return self

    def __repr__(self):
        return f"Config({self._cfg})"
