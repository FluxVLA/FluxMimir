from pathlib import Path

import pytest

from fluxmimir.cli.annotate import main as annotate_main
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
