import argparse
import sys
from pathlib import Path
from typing import Sequence

from mmengine import Config, DictAction


def build_parser(command: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config', type=Path, help='MMEngine Python config')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config values, for example pipeline.runner.workers=2',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load and inspect the plan without executing data operations',
    )
    parser.set_defaults(command=command)
    return parser


def load_config(args: argparse.Namespace, expected_kind: str) -> Config:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'config does not exist: {config_path}')

    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if 'pipeline' not in cfg or not isinstance(cfg.pipeline, dict):
        raise KeyError('config must define a mapping named `pipeline`')

    kind = cfg.pipeline.get('kind')
    if kind != expected_kind:
        raise ValueError(
            f'expected pipeline.kind={expected_kind!r}, got {kind!r}')
    return cfg


def _component_type(value) -> str:
    if isinstance(value, dict):
        return str(value.get('type', '<unspecified>'))
    return '<unspecified>'


def describe_plan(cfg: Config, config_path: Path) -> str:
    pipeline = cfg.pipeline
    lines = [
        f'Config: {config_path.expanduser().resolve()}',
        f'Pipeline kind: {pipeline.kind}',
        f'Runner: {_component_type(pipeline.get("runner", {}))}',
        f'Source: {_component_type(pipeline.get("source", {}))}',
    ]
    stages = pipeline.get('stages', [])
    lines.append('Stages: ' +
                 (', '.join(_component_type(stage)
                            for stage in stages) or '<none>'))
    lines.append(f'Writer: {_component_type(pipeline.get("writer", {}))}')
    validators = pipeline.get('validators', [])
    lines.append('Validators: ' +
                 (', '.join(_component_type(item)
                            for item in validators) or '<none>'))
    return '\n'.join(lines)


def scaffold_main(*,
                  expected_kind: str,
                  description: str,
                  argv: Sequence[str] | None = None) -> int:
    parser = build_parser(expected_kind, description)
    args = parser.parse_args(argv)
    cfg = load_config(args, expected_kind)
    print(describe_plan(cfg, args.config))
    if args.dry_run:
        print('Dry run complete; no data was read or written.')
        return 0

    print(
        'Execution is intentionally unavailable in the initial scaffold. '
        'Use --dry-run to inspect the config.',
        file=sys.stderr,
    )
    return 2
