from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Skill:
    name: str
    path: str
    summary: str = ""
    tags: set[str] = field(default_factory=set)


class SkillRegistry:
    """项目技能目录管理，提供给 DeepAgents 的 skills sources。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def load_from_directory(self, root: str) -> list[Skill]:
        out: list[Skill] = []
        root_path = Path(root)
        if not root_path.exists():
            return out

        for file in root_path.glob("**/SKILL.md"):
            name = file.parent.name
            summary = ""
            try:
                with file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip() and not line.strip().startswith("#"):
                            summary = line.strip()
                            break
            except OSError:
                summary = ""
            skill = Skill(name=name, path=str(file.parent), summary=summary)
            self.register(skill)
            out.append(skill)
        return out

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def to_sources(self, *, names: list[str] | None = None) -> list[str]:
        if names is None:
            return [item.path for item in self._skills.values()]
        return [self._skills[name].path for name in names if name in self._skills]
