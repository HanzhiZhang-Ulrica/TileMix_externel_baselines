# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from importlib import import_module

from .configs.model2path import get_support_models
from .version import VERSION as __version__

_LAZY_EXPORTS = {
    "MInference": (".models_patch", "MInference"),
    "MInferenceConfig": (".minference_configuration", "MInferenceConfig"),
    "minference_patch": (".patch", "minference_patch"),
    "minference_patch_kv_cache_cpu": (".patch", "minference_patch_kv_cache_cpu"),
    "minference_patch_with_kvcompress": (".patch", "minference_patch_with_kvcompress"),
    "patch_hf": (".patch", "patch_hf"),
    "vertical_slash_sparse_attention": (".ops.pit_sparse_flash_attention_v2", "vertical_slash_sparse_attention"),
    "block_sparse_attention": (".ops.block_sparse_flash_attention", "block_sparse_attention"),
    "streaming_forward": (".ops.streaming_kernel", "streaming_forward"),
}

__all__ = [
    "MInference",
    "MInferenceConfig",
    "minference_patch",
    "minference_patch_kv_cache_cpu",
    "minference_patch_with_kvcompress",
    "patch_hf",
    "vertical_slash_sparse_attention",
    "block_sparse_attention",
    "streaming_forward",
    "get_support_models",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))
