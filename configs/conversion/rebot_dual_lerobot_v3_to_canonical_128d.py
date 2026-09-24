raw_root = ('{{$FLUXMIMIR_RAW_ROOT:'
            'datasets/realrobot_rebot_dual_lerobot_v3}}')
output_root = ('{{$FLUXMIMIR_OUTPUT_ROOT:'
               'work_dirs/datasets/realrobot_rebot_dual_canonical_128d}}')

pipeline = dict(
    kind='conversion',
    runner=dict(
        type='PipelineRunner',
        work_dir='work_dirs/conversion/rebot_dual',
        resume=True,
        overwrite=False,
        atomic_episode_commit=True,
    ),
    source=dict(
        type='LeRobotV3Source',
        root=raw_root,
        camera_mapping={
            'cam_high': 'observation.images.cam_front',
            'cam_left_wrist': 'observation.images.cam_wrist_left',
            'cam_right_wrist': 'observation.images.cam_wrist_right',
        },
        state_key='observation.state',
        action_key='action',
        max_episodes=None,
    ),
    embodiment=dict(
        type='ReBotDualAdapter',
        action_semantics='issued_joint_position_target',
        action_source='operator',
        nominal_fps=30,
        missing_dense_value=0.0,
        gripper_min_rad=-4.712389,
        gripper_max_rad=0.0,
    ),
    stages=[
        dict(type='ProbeSourceSchema'),
        dict(type='MapReBotToCanonical128D'),
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
        dataset_id='realrobot_rebot_dual_pick_place',
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
