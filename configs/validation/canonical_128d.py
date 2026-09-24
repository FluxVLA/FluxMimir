dataset_root = '{{$FLUXMIMIR_DATASET_ROOT:datasets/canonical_input}}'

pipeline = dict(
    kind='validation',
    runner=dict(
        type='ValidationRunner',
        work_dir='work_dirs/validation/canonical_128d',
        write_report_only=True,
        mutate_dataset=False,
    ),
    source=dict(
        type='CanonicalDatasetSource',
        root=dataset_root,
        expected_schema_id='agilex_aloha_unified',
        expected_schema_version='2.0.0',
    ),
    stages=[],
    writer=dict(type='JSONValidationReportWriter'),
    validators=[
        dict(type='CanonicalParquetSchemaValidator'),
        dict(type='UnifiedMaskValidator'),
        dict(type='EpisodeIndexValidator'),
        dict(type='ForeignKeyClosureValidator'),
        dict(type='VideoAlignmentValidator'),
        dict(type='MaskedStatisticsValidator'),
        dict(type='ArtifactManifestValidator'),
    ],
)
