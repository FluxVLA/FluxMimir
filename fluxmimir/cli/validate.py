from typing import Sequence

from .common import scaffold_main


def main(argv: Sequence[str] | None = None) -> int:
    return scaffold_main(
        expected_kind='validation',
        description='Inspect a FluxMimir dataset validation plan.',
        argv=argv,
    )
