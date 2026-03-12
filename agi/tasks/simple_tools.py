from __future__ import annotations

from typing import Any

from agi.tasks.define import AskHuman
from agi.agent.tools.stock_market import get_stock
from agi.agent.tools.weather import get_weather_info


def get_time(timezone: str = "UTC") -> str:
    """Return current time in an IANA timezone string."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo(timezone)).isoformat()


def calc(expression: str) -> Any:
    """Evaluate a basic math expression safely."""
    import ast
    import operator

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


simple_tools = [
    AskHuman,
    get_weather_info,
    get_stock,
    get_time,
    calc,
]
