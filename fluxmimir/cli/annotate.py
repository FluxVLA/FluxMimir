from typing import Sequence

from .common import scaffold_main


def main(argv: Sequence[str] | None = None) -> int:
    return scaffold_main(
        expected_kind='annotation',
        description='Inspect a FluxMimir automatic annotation plan.',
        argv=argv,
    )
