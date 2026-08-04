source_root = '{{$FLUXMIMIR_EGO_SOURCE_ROOT:datasets/ego_source}}'
output_root = ('{{$FLUXMIMIR_SYNTHESIS_OUTPUT_ROOT:'
               'work_dirs/datasets/ego_to_robot}}')

pipeline = dict(
    kind='synthesis',
    runner=dict(
        type='PipelineRunner',
        work_dir='work_dirs/synthesis/ego_to_robot',
        resume=True,
        overwrite=False,
        cache_by_source_fingerprint=True,
    ),
    source=dict(
        type='EgoEpisodeSource',
        root=source_root,
        require_hand_keypoints=True,
        require_camera_contract=True,
    ),
    stages=[
        dict(
            type='RetargetHandTrajectory',
            backend=dict(type='MANOHandTrajectoryRetargeter'),
            position_smoothing='savitzky_golay',
            rotation_smoothing='gaussian_weighted_slerp',
        ),
        dict(
            type='SegmentHumanRegion',
            output_key='human_region_mask',
            backend=dict(
                type='SAM3HumanRegionSegmenter',
                checkpoint_env='FLUXMIMIR_SAM3_CHECKPOINT',
                prompts=['human hand', 'human arm'],
            ),
        ),
        dict(
            type='InpaintHumanRegion',
            backend=dict(
                type='ProPainterVideoInpainter',
                checkpoint_env='FLUXMIMIR_PROPAINTER_CHECKPOINT',
            ),
        ),
        dict(
            type='SearchRobotBasePose',
            backend=dict(type='MujocoRobotKinematicsRenderer'),
            objective='representative_frame_ik_feasibility',
        ),
        dict(
            type='RenderRobot',
            outputs=['robot_rgb', 'robot_depth', 'joint_trajectory'],
        ),
        dict(
            type='EstimateMetricSceneDepth',
            backend=dict(
                type='DepthAnythingV3Estimator',
                checkpoint_env='FLUXMIMIR_DEPTH_CHECKPOINT',
            ),
        ),
        dict(
            type='DepthOcclusionComposite',
            rule='robot_depth_le_scene_depth',
        ),
    ],
    writer=dict(
        type='CanonicalDerivedDatasetWriter',
        root=output_root,
        in_place=False,
        retain_intermediate_artifacts=True,
        publish_provenance=True,
    ),
    validators=[
        dict(type='HumanRegionMaskTemporalValidator'),
        dict(type='InpaintingArtifactValidator'),
        dict(type='IKFeasibilityValidator', minimum_ratio=0.95),
        dict(type='DepthFrameAndUnitValidator'),
        dict(type='RobotReprojectionIoUValidator', reject_below=0.5),
        dict(type='OutputFPSValidator'),
    ],
)
