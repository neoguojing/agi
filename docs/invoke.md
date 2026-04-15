invoke()
  ↓
stream()
  ↓
SyncPregelLoop (关键入口)
  ↓
PregelRunner
  ↓
loop.tick()


PregelRunner
    ↓
run_with_retry
    ↓
task.proc.invoke   ⭐⭐⭐（关键入口）
    ↓
node logic
    ↓
return / writes
    ↓
commit
    ↓
put_writes
    ↓
checkpoint


  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/_internal/_runnable.py(473)ainvoke()
-> ret = await self.afunc(*args, **kwargs)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(1317)amodel_node()
-> result = await awrap_model_call_handler(request, _execute_model_async)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/middleware/todo.py(253)awrap_model_call()
-> return await handler(request.override(system_message=new_system_message))
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(361)inner_handler()
-> inner_result = await inner(req, handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/deepagents/middleware/filesystem.py(1126)awrap_model_call()
-> return await handler(request)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(361)inner_handler()
-> inner_result = await inner(req, handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/deepagents/middleware/subagents.py(691)awrap_model_call()
-> return await handler(request.override(system_message=new_system_message))
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(361)inner_handler()
-> inner_result = await inner(req, handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/deepagents/middleware/summarization.py(1022)awrap_model_call()
-> return await handler(request.override(messages=truncated_messages))
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(361)inner_handler()
-> inner_result = await inner(req, handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain_anthropic/middleware/prompt_caching.py(140)awrap_model_ca
ll()
-> return await handler(request)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(361)inner_handler()
-> inner_result = await inner(req, handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/middleware/model_fallback.py(126)awrap_model_call(
)
-> return await handler(request)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(361)inner_handler()
-> inner_result = await inner(req, handler)
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langchain/agents/factory.py(371)composed()
-> outer_result = await outer(request, inner_handler)
> /home/SENSETIME/guoerjun/workspace/src/agi/agi/agent/middlewares/context_middleware.py(65)awrap_model_call()
-> request.runtime.context.set_incremental_messages(request.messages)

updates first
/home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/main.py(3026)astream()
-> async with AsyncPregelLoop(
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/_loop.py(1379)__aenter__()
-> self.updated_channels = self._first(
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/_loop.py(735)_first()
-> updated_channels = apply_writes(
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/_algo.py(306)apply_writes()
-> if channels[chan].update(EMPTY_SEQ) and next_version is not None:
> /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/channels/binop.py(105)update()
-> return False
after
home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/main.py(3127)astream()
-> loop.after_tick()
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/_loop.py(544)after_tick()
-> self.updated_channels = apply_writes(
  /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/pregel/_algo.py(296)apply_writes()
-> if channels[chan].update(vals) and next_version is not None:
> /home/SENSETIME/guoerjun/miniconda3/envs/open-webui/lib/python3.11/site-packages/langgraph/channels/binop.py(104)update()



Graph.invoke / astream
   ↓
PregelLoop.run / atick
   ↓
prepare_next_tasks
   ↓
prepare_single_task
   ↓
_proc_input  ← 从 channel 读取 state（messages 在这里进来）
   ↓
PregelExecutableTask
   ↓
arun_with_retry
   ↓
task.proc.ainvoke
   ↓
amodel_node（model 节点）
   ↓
LLM.generate
   ↓
node / middleware 写入 writes（messages 等）
   ↓
after_tick
   ↓
apply_writes  ← 写回 channel（state 更新核心）
   ↓
channel.update（BinaryOperatorAggregate 等）
   ↓
下一轮

while 有任务:
    执行任务
    收集 writes
    apply_writes 更新 state（channels）
    触发下一轮任务

graph.astream()
  ↓
AsyncPregelLoop(...)   ← 初始化运行时（checkpoint / channels / state）
  ↓
while loop.tick():
    ↓
    runner.atick(...)   ← 执行 tasks（node）
        ↓
        task.proc.ainvoke(...)
            ↓
            LangChain Agent / LLM 调用
            ↓
            node 写入 writes（messages / state）
    ↓
    loop.after_tick()   ← ⭐ 关键点
        ↓
        apply_writes(...)   ← ⭐ 真正更新 state 的地方
            ↓
            channel.update(...)  ← ⭐ 每个 channel 自己决定如何合并


channels结构：
{'messages': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x7fca48980d80>, 'jump_to': 
<langgraph.channels.ephemeral_value.EphemeralValue object at 0x7fca4874e700>, 'structured_response': <langgraph.channels.last_value.LastValue object 
at 0x7fca4874e780>, 'files': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x7fca4874e680>, '_summarization_event': 
<langgraph.channels.last_value.LastValue object at 0x7fca4895e6c0>, 'todos': <langgraph.channels.last_value.LastValue object at 0x7fca48779bc0>, 
'__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7fca4895d900>, '__pregel_tasks': <langgraph.channels.topic.Topic object at
0x7fca48791dc0>, 'branch:to:model': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7fca48791c40>, 'branch:to:tools': 
<langgraph.channels.ephemeral_value.EphemeralValue object at 0x7fca48792040>, 'branch:to:TodoListMiddleware.after_model': 
<langgraph.channels.ephemeral_value.EphemeralValue object at 0x7fca48792240>, 'branch:to:PatchToolCallsMiddleware.before_agent': 
<langgraph.channels.ephemeral_value.EphemeralValue object at 0x7fca48792200>}