from types import SimpleNamespace

from agi.agents_runtime.hitl import build_resume_payload, extract_interrupt_actions


def test_extract_interrupt_actions():
    result = {
        "__interrupt__": [
            SimpleNamespace(
                value={
                    "action_requests": [{"name": "delete_file", "args": {"path": "/tmp/a"}}],
                    "review_configs": [{"action_name": "delete_file", "allowed_decisions": ["approve", "reject"]}],
                }
            )
        ]
    }

    actions = extract_interrupt_actions(result)
    assert len(actions) == 1
    assert actions[0].name == "delete_file"
    assert actions[0].allowed_decisions == ["approve", "reject"]


def test_build_resume_payload():
    payload = build_resume_payload([{"type": "approve"}])
    assert payload == {"decisions": [{"type": "approve"}]}
