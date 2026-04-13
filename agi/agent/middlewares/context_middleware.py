import asyncio
import time
from typing import Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from deepagents.backends.protocol import BackendProtocol

from agi.agent.context import UnifiedContextManager, ContextCompressor
from agi.utils.common import append_to_system_message


class ContextEngineeringMiddleware(AgentMiddleware):
    """
    上下文工程中间件：

    1. 动态注入模型 Prompt
    2. 异步更新用户画像（Semantic / Episodic / Procedural）
    3. 消息压缩
    4. idle-trigger evolution（防抖）
    """

    def __init__(
        self,
        extractor_model,
        backend=None,
        idle_seconds: int = 5,
        evolve_threshold: int = 10,
        **config
    ):
        self.backend = backend
        self.manager = UnifiedContextManager(llm_model=extractor_model)

        self.compressor = ContextCompressor(
            threshold=config.get("threshold", 300),
            storage_dir=config.get("storage_dir", "/compressed_messages"),
            keep_recent=config.get("keep_recent", 10)
        )

        # ===== per-user 状态（关键修复）=====
        self.last_sync_cursor = {}          # user_id -> int
        self.pending_evolution = {}         # user_id -> bool
        self.last_request_time = {}         # user_id -> float

        self.idle_seconds = idle_seconds
        self.evolve_threshold = evolve_threshold

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:

        user_id = request.runtime.context.user_id
        self.last_request_time[user_id] = time.time()

        # =========================
        # 1. load cursor（修复 store bug）
        # =========================
        cursor_obj = await request.runtime.store.aget(user_id, "context_status")
        last_cursor = 0
        if cursor_obj:
            last_cursor = cursor_obj.value.get("last_sync_cursor") or 0

        # =========================
        # 2. 保留原始消息（关键修复）
        # =========================
        original_messages = request.messages
        print("********************",request.state.get("messages"))
        # =========================
        # 3. context injection
        # =========================
        injected_context = await self.manager.get_context(request.runtime)

        # =========================
        # 4. compression
        # =========================
        backend = self._get_backend(request.runtime)
        compressed_messages = await self.compressor.compress(original_messages, backend)

        # =========================
        # 5. override request
        # =========================
        request = request.override(
            messages=compressed_messages,
            system_message=append_to_system_message(
                request.system_message,
                injected_context
            )
        )

        # =========================
        # 6. model call
        # =========================
        self._log_debug_info(injected_context, len(original_messages))
        response = await handler(request)

        # =========================
        # 7. evolution trigger
        # =========================
        current_cursor = len(original_messages)

        if (current_cursor - last_cursor) >= self.evolve_threshold:
            if not self.pending_evolution.get(user_id, False):
                self.pending_evolution[user_id] = True

                asyncio.create_task(
                    self._wait_for_idle_and_evolve(
                        runtime=request.runtime,
                        user_id=user_id,
                        messages=original_messages
                    )
                )

        return response

    async def _wait_for_idle_and_evolve(self, runtime, user_id: str, messages):

        print(f"--- [Evolution] user={user_id} 等待 idle={self.idle_seconds}s ---")

        while True:
            await asyncio.sleep(1)

            last_t = self.last_request_time.get(user_id, 0)
            elapsed = time.time() - last_t

            if elapsed >= self.idle_seconds:
                try:
                    await self._perform_deep_evolution(runtime, user_id, messages)

                    # 更新 cursor（基于 raw messages）
                    self.last_sync_cursor[user_id] = len(messages)

                    await runtime.store.aput(
                        user_id,
                        "context_status",
                        {"last_sync_cursor":len(messages)}
                    )

                    self.pending_evolution[user_id] = False

                    print(f"--- [Evolution] user={user_id} 完成 ---")
                    break

                except Exception as e:
                    self.pending_evolution[user_id] = False
                    print(f"[Evolution Error] user={user_id}: {e}")
                    break

    async def _perform_deep_evolution(self, runtime, user_id: str, messages):
        """
        Semantic / Episodic / Procedural memory update
        """

        last_cursor = self.last_sync_cursor.get(user_id, 0)

        new_messages = messages[last_cursor:]

        if not new_messages:
            return

        await self.manager.update(
            runtime=runtime,
            messages=new_messages
        )

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(
            f"--- [Context Engine] inject={ctx_data} | "
            f"messages={total_count} ---"
        )