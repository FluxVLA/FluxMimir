raw_root = '{{$FLUXMIMIR_RAW_ROOT:datasets/aloha_lerobot_v2}}'
output_root = '{{$FLUXMIMIR_OUTPUT_ROOT:work_dirs/datasets/aloha_v2_128d}}'

pipeline = dict(
    kind='conversion',
    runner=dict(
        type='PipelineRunner',
        work_dir='work_dirs/conversion/agilex_aloha_lerobot_v2',
        resume=True,
        overwrite=False,
        atomic_episode_commit=True,
    ),
    source=dict(
        type='LeRobotV2Source',
        root=raw_root,
        camera_mapping={
            'cam_high': 'observation.images.cam_high',
            'cam_left_wrist': 'observation.images.cam_left_wrist',
            'cam_right_wrist': 'observation.images.cam_right_wrist',
        },
        state_key='observation.state',
        action_key='action',
        eepose_key='observation.eepose',
        max_episodes=None,
    ),
    embodiment=dict(
        type='AgileXAlohaAdapter',
        schema_id='agilex_aloha_unified',
        schema_version='2.0.0',
        layout_id='lingbot_vla2_dexhand32x2_128_v1',
        require_confirmed=[
            'joint_order',
            'action_semantics',
            'timestamp_source',
            'nominal_fps',
            'depth_unit',
            'camera_calibration',
        ],
        missing_dense_value=0.0,
        unknown_target_source=5,
        joint_order='left_arm_gripper_right_arm_gripper',
        # LeRobot v2 export does not prove who issued the stored action.
        action_semantics='unknown',
        nominal_fps=30,
    ),
    stages=[
        dict(type='ProbeSourceSchema'),
        dict(type='MapAlohaToCanonical128D'),
        # Static frame filtering is opt-in. Start with policy='report' and
        # inspect work_dir/static_frames.jsonl before dropping episodes.
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
        dataset_id='realrobot_agilex_aloha_lerobot_v2',
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
