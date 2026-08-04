from pathlib import Path

import pytest
from mmengine import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'conversion': REPO_ROOT / 'configs' / 'conversion' /
    'agilex_aloha_hdf_to_canonical_107d.py',
    'synthesis':
    REPO_ROOT / 'configs' / 'synthesis' / 'qwen_robotmanip_ego_to_robot.py',
    'annotation':
    REPO_ROOT / 'configs' / 'annotation' / 'canonical_auto_annotation.py',
    'validation': REPO_ROOT / 'configs' / 'validation' / 'canonical_107d.py',
}


@pytest.mark.parametrize('expected_kind,config_path', CONFIGS.items())
def test_example_config_has_common_pipeline_shape(expected_kind, config_path):
    cfg = Config.fromfile(config_path)

    assert cfg.pipeline.kind == expected_kind
    assert cfg.pipeline.runner.type
    assert cfg.pipeline.source.type
    assert isinstance(cfg.pipeline.stages, list)
    assert cfg.pipeline.writer.type
    assert isinstance(cfg.pipeline.validators, list)


def test_annotation_config_stores_credential_reference_only():
    cfg = Config.fromfile(CONFIGS['annotation'])
    provider = cfg.pipeline.stages[1].provider

    assert provider.api_key_env == 'ANNOTATION_API_KEY'
    assert 'api_key' not in provider
    assert 'app_secret' not in provider
    assert 'access_token' not in provider
