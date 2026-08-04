# FluxMimir Architecture

## Status

The HDF-to-canonical conversion path is implemented and guarded by an explicit
`--execute` flag. It writes Parquet, optional videos, metadata, masked
statistics, and a content manifest, then reopens the release for validation.
Vision synthesis and annotation providers remain interface scaffolds.

## Responsibility boundaries

The three Flux repositories have separate owners and artifacts:

| Repository | Sole responsibility |
| --- | --- |
| FluxMimir | Offline raw ingestion, canonical conversion, derived visual synthesis, annotation proposals, data QC, and dataset release manifests |
| FluxVLA | Training/evaluation dataset readers and transforms, normalization, models, checkpoints, training, and inference |
| FluxThemis | Simulation/robot rollout, evaluation lifecycle, environments, model clients, and evaluation metrics |

FluxMimir never owns model training or robot rollout. FluxVLA should consume an
immutable, validated FluxMimir release rather than import a converter. FluxMimir
does not load FluxVLA model configs, and FluxThemis does not become a data
conversion dependency.

## Processing model

```text
Physical source
  -> source adapter (discover and decode only)
  -> EpisodeRecord source-neutral IR
  -> embodiment adapter (order, units, frames, validity)
  -> ordered stages (conversion / synthesis / annotation)
  -> transactional writer in a new staging release
  -> independent validators and quality gates
  -> immutable dataset version + content manifest
```

The source adapter cannot infer target semantics from a field name. The
embodiment adapter cannot mark a value valid until order, unit, frame, timing,
and provenance are established. The writer cannot silently repair validation
failures. Raw input is always read-only, existing releases are never overwritten
by default, and incomplete staging output is not a release.

## Package layout

```text
fluxmimir/
  annotation/            typed annotation request/record/provider contracts
  backends/              replaceable external tool/model interfaces
  cli/                   thin config-inspection entry points
  conversion/            source-neutral EpisodeRecord and mapper contracts
  embodiments/           robot-specific semantic adapters
  engines/               registries, builders, stages, and runners
  media/                 streaming media encoder contracts
  schemas/               versioned canonical data contracts
  sources/               physical input discovery/decoding boundaries
  synthesis/ego_to_robot ordered ego-to-robot stage contract
  validators/            structured validation reports
  writers/               transactional release writer contracts
```

Stable cross-package values use dataclasses in `fluxmimir.structures`. Large
arrays and videos remain lazy/source-owned until a stage requests them; they are
not embedded in `EpisodeRef` or control-plane manifests.

## Registry and config contract

FluxMimir follows the FluxVLA/FluxThemis MMEngine convention. Components are
selected with `dict(type='...')` and will register into one of:

- `SOURCES`
- `STAGES`
- `BACKENDS`
- `ANNOTATORS`
- `WRITERS`
- `VALIDATORS`
- `RUNNERS`

Concrete implementations import their registration module from package
`__init__.py`; otherwise config construction cannot discover them. Conversion
components in the example config are executable. Other pipeline kinds still
only validate and describe their plans.

Every config defines one top-level `pipeline` mapping:

```python
pipeline = dict(
    kind='conversion',
    runner=dict(type='PipelineRunner'),
    source=dict(type='AlohaHDF5Source'),
    stages=[dict(type='MapAlohaToCanonical107D')],
    writer=dict(type='CanonicalDatasetWriter'),
    validators=[dict(type='CanonicalParquetSchemaValidator')],
)
```

Allowed initial kinds are `conversion`, `synthesis`, `annotation`, and
`validation`. Runtime implementations should preserve stage-level resume/cache
identity, emit structured status and reason codes, and record sanitized
provenance for every artifact.

## Domain boundaries

### Conversion

HDF discovery and physical decoding belong to `sources`; left/right ordering,
units, target provenance, and camera frames belong to `embodiments`; the exact
12-column/107D contract belongs to `schemas`; encoding belongs to `media`; and
Parquet/video/metadata publication belongs to `writers`.

### Ego-to-robot synthesis

The synthesis path is an ordered, replaceable backend graph: retarget hand
trajectory, segment the human region, temporally inpaint it, search a robot base
pose, solve/render the robot, estimate metric scene depth, then composite with
depth-aware occlusion. Visual `human_region_mask[T,H,W]` and training
`action_validity_mask[T,D]` are different contracts and may not share an
ambiguous field name.

### Annotation

Annotation providers receive a versioned request and must return JSON validated
against a versioned schema. Provider output remains a proposal with confidence,
provenance, cost, and review status. Credentials are resolved inside the
provider boundary from an environment-variable name; resolved values never
enter configs, logs, manifests, cache keys, or records.

## Validation and publication

Validation is independent from writing. A future release gate must cover
physical schema, shapes/dtypes, finite values, explicit masks, time/video
alignment, task/subtask foreign keys, mask-aware statistics, geometry and depth
contracts, annotation schemas, quality metrics, and the complete artifact hash
closure. A browsable dataset is not necessarily `training_ready`; unresolved
semantic blockers keep that flag false.

See [data_contract.md](data_contract.md), [conversion.md](conversion.md),
[synthesis.md](synthesis.md), [annotation.md](annotation.md), and
[security.md](security.md) for the detailed contracts.
