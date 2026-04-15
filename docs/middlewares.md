你可以看到明确的 node 注入：

graph.add_node(f"{m.name}.before_agent", ...)
graph.add_node(f"{m.name}.before_model", ...)
graph.add_node(f"{m.name}.after_model", ...)
graph.add_node(f"{m.name}.after_agent", ...)

👉 这些 middleware 会变成：

before_agent → before_model → model → after_model → after_agent

作为函数包装器（hook）

两类 wrapper：

✔ tool call wrapper
wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)
awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)
✔ model call wrapper
wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)
awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

👉 这些不会进入 graph，而是：

👉 包裹 model/tool 执行函数

_chain_model_call_handlers(handlers)

👉 把多个 wrap_model_call middleware：

[h1, h2, h3]

组合成一个：

h1(h2(h3(model_call)))
一句话本质

把多个 middleware 包装成一个“嵌套调用链”，并正确收集 Command