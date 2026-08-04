raw_root = ('{{$FLUXMIMIR_RAW_ROOT:'
            '/mnt/data/oss/limx_mani_data/raw_data/'
            'RealRobot_AgileX_aloha}}')
output_root = ('{{$FLUXMIMIR_OUTPUT_ROOT:'
               'work_dirs/datasets/'
               'realrobot_agilex_aloha_canonical_107d}}')

pipeline = dict(
    kind='conversion',
    runner=dict(
        type='PipelineRunner',
        work_dir='work_dirs/conversion/agilex_aloha',
        resume=True,
        overwrite=False,
        atomic_episode_commit=True,
    ),
    source=dict(
        type='AlohaHDF5Source',
        root=raw_root,
        pattern='**/episode_*.hdf5',
        streaming=True,
        cameras=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        depth_group='observations/images_depth',
        # Set to a small integer for a smoke conversion before a full run.
        max_episodes=None,
    ),
    embodiment=dict(
        type='AgileXAlohaAdapter',
        schema_id='agilex_aloha_unified',
        schema_version='1.1.0',
        layout_id='lingbot_vla2_dexhand32x2_107_v1',
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
        # Raw /action is not made trainable until its controller semantics are
        # explicitly confirmed. Use issued_joint_position_target only with
        # verified provenance and time alignment.
        action_semantics='unknown',
        nominal_fps=30,
    ),
    stages=[
        dict(type='ProbeHDF5Schema', scope='all_episodes'),
        dict(type='MapAlohaToCanonical107D'),
        dict(
            type='EncodeEpisodeVideos',
            codec='h264',
            fps=30,
            frame_count_must_match=True,
        ),
        dict(type='ComputeMaskedEpisodeStatistics'),
        dict(type='BuildCanonicalMetadata'),
    ],
    writer=dict(
        type='CanonicalDatasetWriter',
        root=output_root,
        dataset_id='realrobot_agilex_aloha',
        dataset_version='0.1.0',
        chunks_size=1000,
        publish_manifest=True,
        in_place=False,
    ),
    validators=[
        dict(type='SourceSemanticsValidator'),
        dict(type='CanonicalParquetSchemaValidator'),
        dict(type='ForeignKeyClosureValidator'),
        dict(type='VideoAlignmentValidator'),
        dict(type='MaskedStatisticsValidator'),
        dict(type='ArtifactManifestValidator'),
    ],
    references=dict(
        canonical_example=('/root/projects/FluxVLA/datasets/'
                           'RealRobot_AgileX_aloha_lerobot/'
                           'example_canonical_107d_v3_1'),
        legacy_tools='/root/projects/FluxVLA/tools',
    ),
)
