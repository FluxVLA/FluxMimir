from fluxmimir.engines import SOURCES, build_source_from_cfg


def test_source_registry_builds_configured_component():

    @SOURCES.register_module(name='UnitTestSource', force=True)
    class UnitTestSource:

        def __init__(self, value):
            self.value = value

    source = build_source_from_cfg(dict(type='UnitTestSource', value=7))

    assert source.value == 7
    assert SOURCES.scope == 'fluxmimir'
