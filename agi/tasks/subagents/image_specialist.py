from __future__ import annotations

from agi.tasks.runtime.task_factory import TASK_IMAGE_GEN, TASK_MULTI_MODEL, TaskFactory


image_gen_tool = TaskFactory.create_task(TASK_IMAGE_GEN).as_tool(
    name="image_gen",
    description="Generate images from text prompts.",
)
image_gen_tool.return_direct = True

image_recog_tool = TaskFactory.create_task(TASK_MULTI_MODEL).as_tool(
    name="image_recog",
    description="Recognize and describe image content.",
)
image_recog_tool.return_direct = True


image_subagent = {
    "name": "image_specialist",
    "description": "Use this for image generation, editing, and vision understanding.",
    "system_prompt": "You are an image specialist. Use visual tools and return concise outputs.",
    "tools": [image_gen_tool, image_recog_tool],
}
