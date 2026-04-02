# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module

_LAZY_EXPORTS = {
    "disable_hf_flash_attention_check": (".modules.patch", "disable_hf_flash_attention_check"),
    "get_config_example": (".modules.patch", "get_config_example"),
    "patch_model": (".modules.patch", "patch_model"),
    "flex_prefill_attention": (".ops.flex_prefill_attention", "flex_prefill_attention"),
}

__all__ = [
    "flex_prefill_attention",
    "patch_model",
    "get_config_example",
    "disable_hf_flash_attention_check",
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
