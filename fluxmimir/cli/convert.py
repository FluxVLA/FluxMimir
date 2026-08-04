import json
import sys
from typing import Sequence

from fluxmimir.engines import build_runner_from_cfg

from .common import build_parser, describe_plan, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(
        'conversion',
        'Convert episode HDF5 files into canonical 107D Parquet datasets.',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Explicitly execute the conversion plan.',
    )
    args = parser.parse_args(argv)
    cfg = load_config(args, 'conversion')
    print(describe_plan(cfg, args.config))
    if args.dry_run:
        print('Dry run complete; no data was read or written.')
        return 0
    if not args.execute:
        print(
            'Execution is intentionally unavailable without explicit '
            '--execute confirmation.',
            file=sys.stderr,
        )
        return 2

    pipeline = cfg.pipeline
    runner = build_runner_from_cfg(
        pipeline.runner,
        default_args={
            'source': pipeline.source,
            'embodiment': pipeline.embodiment,
            'stages': pipeline.stages,
            'writer': pipeline.writer,
            'validators': pipeline.get('validators', []),
        },
    )
    try:
        result = runner.run()
    finally:
        runner.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
