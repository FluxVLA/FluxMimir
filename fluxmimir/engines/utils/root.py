from mmengine.registry import Registry

SOURCES = Registry('sources', scope='fluxmimir')
STAGES = Registry('stages', scope='fluxmimir')
EMBODIMENTS = Registry('embodiments', scope='fluxmimir')
BACKENDS = Registry('backends', scope='fluxmimir')
ANNOTATORS = Registry('annotators', scope='fluxmimir')
WRITERS = Registry('writers', scope='fluxmimir')
VALIDATORS = Registry('validators', scope='fluxmimir')
RUNNERS = Registry('runners', scope='fluxmimir')
