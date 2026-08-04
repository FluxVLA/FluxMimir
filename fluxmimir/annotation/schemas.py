TASK_SUBTASK_SCHEMA_ID = 'fluxmimir.task_subtask.v1'
ECOT_SCHEMA_ID = 'fluxmimir.ecot.v1'

TASK_SUBTASK_JSON_SCHEMA = {
    'type': 'object',
    'required': ['task', 'subtasks'],
    'properties': {
        'task': {
            'type': 'string'
        },
        'subtasks': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['start_frame', 'end_frame', 'instruction'],
                'properties': {
                    'start_frame': {
                        'type': 'integer',
                        'minimum': 0
                    },
                    'end_frame': {
                        'type': 'integer',
                        'minimum': 0
                    },
                    'instruction': {
                        'type': 'string'
                    },
                },
            },
        },
    },
}

ECOT_JSON_SCHEMA = {
    'type':
    'object',
    'required': [
        'scene_description',
        'task_progress_assessment',
        'next_atomic_action',
    ],
    'properties': {
        'scene_description': {
            'type': 'string'
        },
        'task_progress_assessment': {
            'type': 'object',
            'required': ['status', 'reason'],
            'properties': {
                'status': {
                    'enum': ['complete', 'not_yet_complete']
                },
                'reason': {
                    'type': 'string'
                },
            },
        },
        'next_atomic_action': {
            'type': 'string'
        },
    },
}
