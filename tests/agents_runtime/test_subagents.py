from agi.agents_runtime.subagents import SubAgentSpec, build_default_subagents
from agi.agents_runtime.types import AgentRuntimeConfig


def test_subagent_spec_to_deepagents_payload():
    spec = SubAgentSpec(
        name="researcher",
        description="research",
        system_prompt="do research",
        tools=[],
        model="openai:gpt-5",
        skills=["/skills/research"],
    )

    payload = spec.to_deepagents()
    assert payload["name"] == "researcher"
    assert payload["model"] == "openai:gpt-5"
    assert payload["skills"] == ["/skills/research"]


def test_build_default_subagents():
    out = build_default_subagents(model="openai:gpt-5", skill_sources=["/skills/main"], code_tools=[])
    assert len(out) == 3
    names = {x["name"] for x in out}
    assert "knowledge-researcher" in names
    assert "multimodal-worker" in names
    assert "code-engineer" in names


def test_runtime_config_has_subagent_fields():
    config = AgentRuntimeConfig()
    assert config.use_default_subagents is True
    assert config.subagents == []
