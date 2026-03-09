from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_SKILL_MD_BYTES = 10 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 1024


@dataclass(slots=True)
class Skill:
    name: str
    path: str
    summary: str = ""
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillRegistry:
    """插件式技能管理：支持加载、注册、更新、删除、优先级源管理。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._sources: list[str] = []

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        if name not in self._skills:
            return False
        del self._skills[name]
        return True

    def update(self, name: str, **changes: Any) -> bool:
        if name not in self._skills:
            return False
        skill = self._skills[name]
        for key, value in changes.items():
            if hasattr(skill, key):
                setattr(skill, key, value)
        return True

    def load_from_directory(self, root: str) -> list[Skill]:
        out: list[Skill] = []
        root_path = Path(root)
        if not root_path.exists():
            return out

        self.add_source(str(root_path))

        for file in root_path.glob("**/SKILL.md"):
            if file.stat().st_size > MAX_SKILL_MD_BYTES:
                continue

            info = self._parse_skill_md(file)
            if not info["name"]:
                info["name"] = file.parent.name
            skill = Skill(
                name=info["name"],
                path=str(file.parent),
                summary=info["description"],
                metadata=info,
            )
            self.register(skill)
            out.append(skill)
        return out

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def add_source(self, source: str) -> None:
        self._sources = [x for x in self._sources if x != source] + [source]

    def remove_source(self, source: str) -> bool:
        if source not in self._sources:
            return False
        self._sources = [x for x in self._sources if x != source]
        return True

    def list_sources(self) -> list[str]:
        return list(self._sources)

    def to_sources(self, *, names: list[str] | None = None) -> list[str]:
        if names is None:
            return self.list_sources() if self._sources else [item.path for item in self._skills.values()]
        return [self._skills[name].path for name in names if name in self._skills]

    def create_plugin_skill(self, root: str, name: str, description: str, body: str, *, overwrite: bool = False) -> str:
        """创建一个技能插件目录与 SKILL.md。"""
        skill_dir = Path(root) / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists() and not overwrite:
            raise FileExistsError(f"Skill already exists: {skill_md}")

        frontmatter = (
            "---\n"
            f"name: {name}\n"
            f"description: {description[:MAX_DESCRIPTION_CHARS]}\n"
            "---\n\n"
        )
        skill_md.write_text(frontmatter + body.strip() + "\n", encoding="utf-8")
        self.load_from_directory(root)
        return str(skill_dir)

    def delete_plugin_skill(self, root: str, name: str) -> bool:
        skill_dir = Path(root) / name
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return False

        skill_md.unlink(missing_ok=True)
        # remove directory only when empty
        try:
            skill_dir.rmdir()
        except OSError:
            pass
        self.unregister(name)
        return True

    def update_plugin_skill(
        self,
        root: str,
        name: str,
        *,
        description: str | None = None,
        body: str | None = None,
    ) -> bool:
        skill_md = Path(root) / name / "SKILL.md"
        if not skill_md.exists():
            return False

        parsed = self._parse_skill_md(skill_md)
        final_description = (description if description is not None else parsed.get("description", ""))[:MAX_DESCRIPTION_CHARS]
        final_body = body if body is not None else parsed.get("body", "")

        frontmatter = (
            "---\n"
            f"name: {name}\n"
            f"description: {final_description}\n"
            "---\n\n"
        )
        skill_md.write_text(frontmatter + final_body.strip() + "\n", encoding="utf-8")
        self.load_from_directory(str(Path(root)))
        return True

    @staticmethod
    def _parse_skill_md(path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        result: dict[str, Any] = {"name": "", "description": "", "body": text}
        if not text.startswith("---"):
            # fallback: first non-heading line as summary
            for line in text.splitlines():
                striped = line.strip()
                if striped and not striped.startswith("#"):
                    result["description"] = striped[:MAX_DESCRIPTION_CHARS]
                    break
            return result

        parts = text.split("---", 2)
        if len(parts) < 3:
            return result

        frontmatter = parts[1]
        body = parts[2]
        result["body"] = body

        for raw_line in frontmatter.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "name":
                result["name"] = value
            elif key == "description":
                result["description"] = value[:MAX_DESCRIPTION_CHARS]
            else:
                result[key] = value

        return result
