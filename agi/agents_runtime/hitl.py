from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class InterruptAction:
    name: str
    args: dict[str, Any]
    allowed_decisions: list[str]


def extract_interrupt_actions(result: dict[str, Any]) -> list[InterruptAction]:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return []

    value = interrupts[0].value
    action_requests = value.get("action_requests", [])
    review_configs = value.get("review_configs", [])
    config_map = {cfg.get("action_name"): cfg for cfg in review_configs}

    out: list[InterruptAction] = []
    for action in action_requests:
        cfg = config_map.get(action.get("name"), {})
        out.append(
            InterruptAction(
                name=action.get("name", ""),
                args=action.get("args", {}),
                allowed_decisions=cfg.get("allowed_decisions", ["approve", "edit", "reject"]),
            )
        )
    return out


def build_resume_payload(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"decisions": decisions}
