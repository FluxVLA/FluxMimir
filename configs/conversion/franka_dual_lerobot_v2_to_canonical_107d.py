raw_root = '{{$FLUXMIMIR_RAW_ROOT:datasets/franka_dual_lerobot_v2.1}}'
output_root = '{{$FLUXMIMIR_OUTPUT_ROOT:work_dirs/datasets/franka_dual_107d}}'

pipeline = dict(
    kind='conversion',
    runner=dict(
        type='PipelineRunner',
        work_dir='work_dirs/conversion/franka_dual',
        resume=True,
        overwrite=False,
        atomic_episode_commit=True,
    ),
    source=dict(
        type='LeRobotV2Source',
        root=raw_root,
        camera_mapping={
            'cam_high': 'observation.images.cam_front',
            'cam_left_wrist': 'observation.images.cam_wrist_left',
            'cam_right_wrist': 'observation.images.cam_wrist_right',
        },
        state_key='observation.state',
        action_key='action',
        eepose_key='observation.eepose',
        max_episodes=None,
    ),
    embodiment=dict(
        type='FrankaDualAdapter',
        action_semantics='unknown',
        nominal_fps=30,
        gripper_min_m=-0.001,
    ),
    stages=[
        dict(type='ProbeSourceSchema'),
        dict(type='MapFrankaToCanonical107D'),
        dict(
            type='FilterStaticFrames',
            enabled=False,
            policy='report',
            static_frame_threshold=50,
            signals=('observation', 'target'),
            groups=('arm_joint_position', 'gripper_position'),
            step_threshold=dict(
                arm_joint_position=1.0e-4, gripper_position=1.5e-4),
            audit_name='static_frames.jsonl',
        ),
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
        dataset_id='realrobot_franka_dual_lerobot_v2',
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
