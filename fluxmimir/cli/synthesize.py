from typing import Sequence

from .common import scaffold_main


def main(argv: Sequence[str] | None = None) -> int:
    return scaffold_main(
        expected_kind='synthesis',
        description='Inspect a FluxMimir ego-to-robot synthesis plan.',
        argv=argv,
    )
