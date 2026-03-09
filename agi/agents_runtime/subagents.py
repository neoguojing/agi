from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class SubAgentSpec:
    name: str
    description: str
    system_prompt: str
    tools: list[Callable[..., Any]] = field(default_factory=list)
    model: str | None = None
    skills: list[str] = field(default_factory=list)

    def to_deepagents(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
        }
        if self.model:
            payload["model"] = self.model
        if self.skills:
            payload["skills"] = self.skills
        return payload


def build_default_subagents(
    *,
    model: str | None = None,
    skill_sources: list[str] | None = None,
    code_tools: list[Callable[..., Any]] | None = None,
) -> list[dict[str, Any]]:
    """默认子代理：用于知识检索、多模态处理与代码工程隔离执行。"""
    shared_skills = skill_sources or []

    knowledge_researcher = SubAgentSpec(
        name="knowledge-researcher",
        description="用于复杂知识库检索、交叉对比和结果压缩总结。",
        system_prompt=(
            "你是检索子代理。优先完成：1) 提炼检索查询；2) 调用可用检索工具；"
            "3) 输出精简结论、关键证据和来源。"
            "不要返回原始大段中间结果，最终输出控制在 300 字以内。"
        ),
        tools=[],
        model=model,
        skills=shared_skills,
    )

    multimodal_worker = SubAgentSpec(
        name="multimodal-worker",
        description="用于图像/音频等多模态任务的执行与结果摘要。",
        system_prompt=(
            "你是多模态子代理。只关注当前任务，调用工具完成图像生成、识别、编辑或音频处理。"
            "最终返回：任务结果摘要、关键参数、后续建议。"
            "禁止输出冗长中间日志。"
        ),
        tools=[],
        model=model,
        skills=shared_skills,
    )

    code_engineer = SubAgentSpec(
        name="code-engineer",
        description="专用于代码工程任务：搭建项目、执行测试、构建镜像、产出工件。",
        system_prompt=(
            "你是代码工程子代理，运行在隔离 sandbox 工具模式。"
            "优先使用 sandbox_execute/sandbox_upload_file/sandbox_download_file 完成任务。"
            "对高风险操作给出明确说明；输出应包含：改动摘要、测试结果、产物路径。"
        ),
        tools=code_tools or [],
        model=model,
        skills=shared_skills,
    )

    return [
        knowledge_researcher.to_deepagents(),
        multimodal_worker.to_deepagents(),
        code_engineer.to_deepagents(),
    ]
