from typing import Any, Mapping

from mmengine.config import Config, ConfigDict
from mmengine.registry import Registry

from .root import (ANNOTATORS, BACKENDS, EMBODIMENTS, RUNNERS, SOURCES, STAGES,
                   VALIDATORS, WRITERS)

ConfigLike = dict | ConfigDict | Config


def _build(cfg: ConfigLike,
           registry: Registry,
           default_args: Mapping[str, Any] | None = None) -> Any:
    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(f'cfg must be a config mapping, got {type(cfg)}')
    return registry.build(cfg, default_args=default_args)


def build_source_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, SOURCES, default_args)


def build_stage_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, STAGES, default_args)


def build_embodiment_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, EMBODIMENTS, default_args)


def build_backend_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, BACKENDS, default_args)


def build_annotator_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, ANNOTATORS, default_args)


def build_writer_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, WRITERS, default_args)


def build_validator_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, VALIDATORS, default_args)


def build_runner_from_cfg(cfg: ConfigLike, default_args=None) -> Any:
    return _build(cfg, RUNNERS, default_args)
