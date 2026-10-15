import argparse
from pathlib import Path

import pytest

from fluxmimir.cli.annotate import main as annotate_main
from fluxmimir.cli.common import load_config
from fluxmimir.cli.convert import main as convert_main
from fluxmimir.cli.synthesize import main as synthesize_main
from fluxmimir.cli.validate import main as validate_main

REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = (
    (
        convert_main,
        REPO_ROOT / 'configs' / 'conversion' /
        'agilex_aloha_hdf_to_canonical_128d.py',
        'conversion',
    ),
    (
        synthesize_main,
        REPO_ROOT / 'configs' / 'synthesis' /
        'qwen_robotmanip_ego_to_robot.py',
        'synthesis',
    ),
    (
        annotate_main,
        REPO_ROOT / 'configs' / 'annotation' / 'canonical_auto_annotation.py',
        'annotation',
    ),
    (
        validate_main,
        REPO_ROOT / 'configs' / 'validation' / 'canonical_128d.py',
        'validation',
    ),
)


@pytest.mark.parametrize('entrypoint,config_path,kind', CASES)
def test_dry_run_only_describes_plan(entrypoint, config_path, kind, capsys):
    assert entrypoint([str(config_path), '--dry-run']) == 0

    output = capsys.readouterr()
    assert f'Pipeline kind: {kind}' in output.out
    assert 'Dry run complete; no data was read or written.' in output.out
    assert output.err == ''


def test_execution_is_explicitly_unavailable(capsys):
    config_path = CASES[0][1]

    assert convert_main([str(config_path)]) == 2
    captured = capsys.readouterr()
    assert 'intentionally unavailable' in captured.err


def _conversion_args(config_path, **overrides):
    args = dict(
        config=config_path,
        cfg_options=None,
        raw_root=None,
        output_root=None,
        dry_run=True,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_root_flags_override_source_and_writer(tmp_path):
    cfg = load_config(
        _conversion_args(
            CASES[0][1],
            raw_root=str(tmp_path / 'raw'),
            output_root=str(tmp_path / 'out'),
        ), 'conversion')

    assert cfg.pipeline.source.root == str(tmp_path / 'raw')
    assert cfg.pipeline.writer.root == str(tmp_path / 'out')


def test_root_flags_win_over_cfg_options():
    cfg = load_config(
        _conversion_args(
            CASES[0][1],
            cfg_options={'pipeline.source.root': '/from/cfg_options'},
            raw_root='/from/raw_root_flag',
        ), 'conversion')

    assert cfg.pipeline.source.root == '/from/raw_root_flag'


def test_output_root_is_rejected_without_a_writer_root():
    with pytest.raises(ValueError, match='--output-root'):
        validate_main([
            str(CASES[3][1]),
            '--output-root',
            '/tmp/validation_report',
            '--dry-run',
        ])
