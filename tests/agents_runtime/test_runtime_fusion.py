import pytest

pytest.importorskip("langchain_core")

from agi.agents_runtime.knowledge import KnowledgeChunk, KnowledgeFusionService
from agi.agents_runtime.messages import MediaInput, create_multimodal_human_message, message_to_payload
from agi.agents_runtime.multimodal import Modality, MultiModalRequest, MultiModalRouter
from agi.agents_runtime.skills import SkillRegistry
from agi.agents_runtime.tools import ToolRegistry, ToolSpec


def test_multimodal_router_auto_switch():
    router = MultiModalRouter()

    assert router.route(MultiModalRequest(text="生成图片 一只猫")).modality == Modality.IMAGE_GENERATE
    assert router.route(MultiModalRequest(text="请识别图里内容", image="a.png")).modality == Modality.IMAGE_UNDERSTAND
    assert router.route(MultiModalRequest(text="请修改这个图", image="a.png")).modality == Modality.IMAGE_EDIT
    assert router.route(MultiModalRequest(audio="a.wav", target="text")).modality == Modality.AUDIO_TRANSCRIBE
    assert router.route(MultiModalRequest(text="看图", image_base64="abc")).modality == Modality.IMAGE_UNDERSTAND


def test_multimodal_human_message_blocks():
    msg = create_multimodal_human_message(
        text="describe image",
        image=MediaInput(url="https://img.local/cat.png", mime_type="image/png"),
        audio=MediaInput(base64="YWJj", mime_type="audio/wav"),
    )
    payload = message_to_payload(msg)

    assert payload["role"] == "user"
    assert isinstance(payload["content"], list)
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][1]["type"] == "image"
    assert payload["content"][2]["type"] == "audio"


def test_knowledge_inject_to_messages():
    chunks = [KnowledgeChunk(content="doc-a", source="kb1"), KnowledgeChunk(content="doc-b", source="kb2")]
    messages = [{"role": "user", "content": "hello"}]

    out = KnowledgeFusionService.inject_to_messages(messages, chunks)

    assert out[0]["role"] == "system"
    assert "doc-a" in out[0]["content"]
    assert out[1] == messages[0]


def test_tool_registry_filtering_and_export():
    registry = ToolRegistry()

    def t1():
        return "ok"

    def t2():
        return "danger"

    registry.register(ToolSpec(name="read", description="read", handler=t1, modality_tags={"text"}, safety_class="normal"))
    registry.register(ToolSpec(name="exec", description="exec", handler=t2, modality_tags={"text", "code"}, safety_class="high"))

    filtered = registry.list_tools(tags={"code"}, allow_safety={"high"})
    assert len(filtered) == 1
    assert filtered[0].name == "exec"

    exported = registry.to_deepagents_tools(tags={"text"}, allow_safety={"normal"})
    assert exported == [t1]


def test_skill_registry_loads_sources(tmp_path):
    skill_dir = tmp_path / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# writer\n\nrewrite docs\n")

    registry = SkillRegistry()
    loaded = registry.load_from_directory(str(tmp_path / "skills"))

    assert len(loaded) == 1
    assert registry.to_sources() == [str(skill_dir)]
