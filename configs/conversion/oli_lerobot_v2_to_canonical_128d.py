raw_root = '{{$FLUXMIMIR_RAW_ROOT:datasets/oli_lerobot_v2.1}}'
output_root = '{{$FLUXMIMIR_OUTPUT_ROOT:work_dirs/datasets/oli_128d}}'

pipeline = dict(
    kind='conversion',
    runner=dict(
        type='PipelineRunner',
        work_dir='work_dirs/conversion/oli',
        resume=True,
        overwrite=False,
        atomic_episode_commit=True,
    ),
    source=dict(
        type='LeRobotV2Source',
        root=raw_root,
        camera_mapping={
            'head': 'observation.images.head',
            'left_wrist': 'observation.images.left_wrist',
        },
        state_key='observation.state',
        action_key='action',
        eepose_key=None,
        max_episodes=None,
    ),
    embodiment=dict(
        type='OliAdapter',
        action_semantics='issued_joint_position_target',
        action_source='operator',
        nominal_fps=30,
        missing_dense_value=0.0,
    ),
    stages=[
        dict(type='ProbeSourceSchema'),
        dict(type='MapOliToCanonical128D'),
        dict(
            type='EncodeEpisodeVideos',
            codec='source',
            fps=30,
            frame_count_must_match=True,
            mode='copy'),
        dict(type='ComputeMaskedEpisodeStatistics'),
        dict(type='BuildCanonicalMetadata'),
    ],
    writer=dict(
        type='CanonicalDatasetWriter',
        root=output_root,
        dataset_id='realrobot_oli_basket',
        dataset_version='0.1.0',
        chunks_size=1000,
        publish_manifest=True,
        in_place=False,
    ),
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
