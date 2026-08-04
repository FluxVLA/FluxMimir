from fluxmimir.annotation import (ECOT_JSON_SCHEMA, TASK_SUBTASK_JSON_SCHEMA,
                                  AnnotationRequest)


def test_annotation_request_requires_idempotent_identity():
    request = AnnotationRequest(
        request_id='content-schema-prompt-model-hash',
        episode_uid='episode-1',
        schema_id='fluxmimir.task_subtask.v1',
        media_uris=('artifact://episode-1/cam-high', ),
    )

    assert request.privileged_context == {}


def test_built_in_annotation_schemas_are_structured():
    assert TASK_SUBTASK_JSON_SCHEMA['required'] == ['task', 'subtasks']
    assert ECOT_JSON_SCHEMA['required'] == [
        'scene_description',
        'task_progress_assessment',
        'next_atomic_action',
    ]
