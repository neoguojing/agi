| 分类                 | 项                                | 说明                                                                    |
| ------------------ | -------------------------------- | --------------------------------------------------------------------- |
| **Middleware**     | SkillsMiddleware                 | 从 backend 加载 Skills，并将技能信息注入到 **System Prompt**                       |
| **核心模式**           | Progressive Disclosure           | Prompt 只提供 **skill metadata**，需要时再读取 `SKILL.md`                       |
| **Skill 存储结构**     | 目录结构                             | `/skills/{skill-name}/SKILL.md`                                       |
|                    | SKILL.md                         | 必须包含 **YAML frontmatter + Markdown instructions**                     |
| **Skill Metadata** | name                             | Skill 名称（≤64字符，小写字母数字+`-`）                                            |
|                    | description                      | Skill 功能描述（≤1024字符）                                                   |
|                    | path                             | SKILL.md 在 backend 中的路径                                               |
|                    | license                          | Skill license（可选）                                                     |
|                    | compatibility                    | 兼容环境说明（可选）                                                            |
|                    | metadata                         | 扩展元数据 `dict[str,str]`                                                 |
|                    | allowed_tools                    | 推荐使用的工具                                                               |
| **SkillsState**    | skills_metadata                  | 已加载的 SkillMetadata 列表（PrivateStateAttr，不传播给父 Agent）                   |
| **Skill Source**   | sources                          | Skills 目录列表，例如 `["/skills/base/","/skills/user/","/skills/project/"]` |
| **覆盖规则**           | last one wins                    | 后面的 source 优先级更高                                                      |
| **Backend 抽象**     | BackendProtocol                  | 统一访问技能存储（不直接访问文件系统）                                                   |
| **Backend接口**      | ls_info / als_info               | 列出目录                                                                  |
|                    | download_files / adownload_files | 下载 SKILL.md                                                           |
| **加载流程**           | before_agent                     | agent 启动时加载 skill metadata                                            |
|                    | _list_skills                     | 扫描 source 下的 skill 目录                                                 |
|                    | download SKILL.md                | 读取 skill 文档                                                           |
|                    | parse YAML                       | 解析 frontmatter                                                        |
|                    | 生成 SkillMetadata                 | 存入 `state.skills_metadata`                                            |
| **Prompt 注入**      | modify_request                   | 构建 Skills System Prompt                                               |
|                    | wrap_model_call                  | 在模型调用前注入 Prompt                                                       |
| **Prompt内容**       | skills_locations                 | 显示技能来源路径                                                              |
|                    | skills_list                      | 列出 skill 名称 + 描述 + path                                               |
| **LLM 使用方式**       | Step1                            | 识别 task 是否匹配某个 skill                                                  |
|                    | Step2                            | 读取对应 `SKILL.md`                                                       |
|                    | Step3                            | 按 skill workflow 执行                                                   |
| **安全限制**           | SKILL.md 最大                      | 10MB                                                                  |
|                    | description 最大                   | 1024 字符                                                               |
| **一句话总结**          | SkillsMiddleware                 | **加载 Skill Metadata → 注入 Prompt → LLM 按需读取 Skill 文档**                 |


---
name: web-research
description: Structured workflow for conducting web research
license: MIT
compatibility: Python 3.10+
allowed-tools: browser search
metadata:
  author: research-team
  version: "1.0"
---

# Web Research Skill

## When to Use
Use this skill when the user asks to research a topic.

## Workflow

1. Identify research questions
2. Search the web
3. Collect sources
4. Summarize findings

## Example

User: "Research quantum computing progress in 2025"

Steps:
- Search web sources
- Compare articles
- Extract key developments


| 字段            | 必须 | 类型     | 说明       |
| ------------- | -- | ------ | -------- |
| name          | ✔  | string | skill 名称 |
| description   | ✔  | string | skill 描述 |
| license       | ✘  | string | license  |
| compatibility | ✘  | string | 运行环境     |
| allowed-tools | ✘  | string | 推荐工具     |
| metadata      | ✘  | dict   | 扩展信息     |


from deepagents.middleware.skills import SkillsMiddleware
from deepagents.backends.filesystem import FilesystemBackend

backend = FilesystemBackend(root_dir="/skills")

middleware = SkillsMiddleware(
    backend=backend,
    sources=[
        "/skills/base/",
        "/skills/user/",
        "/skills/project/"
    ]
)