from agi.agents_runtime.skills import SkillRegistry


def test_skill_plugin_crud(tmp_path):
    root = tmp_path / "skills"
    registry = SkillRegistry()

    created = registry.create_plugin_skill(
        str(root),
        name="kb-research",
        description="research docs",
        body="# kb-research\n\nuse retriever\n",
    )
    assert created.endswith("kb-research")
    assert registry.get("kb-research") is not None

    updated = registry.update_plugin_skill(str(root), "kb-research", description="research docs v2")
    assert updated is True
    assert registry.get("kb-research").summary == "research docs v2"

    removed = registry.delete_plugin_skill(str(root), "kb-research")
    assert removed is True
    assert registry.get("kb-research") is None


def test_skill_sources_precedence_last_wins():
    registry = SkillRegistry()
    registry.add_source("/skills/user")
    registry.add_source("/skills/project")
    registry.add_source("/skills/user")

    # re-adding source moves it to the end: last wins semantics
    assert registry.to_sources() == ["/skills/project", "/skills/user"]
