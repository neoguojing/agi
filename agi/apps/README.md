# agi.apps：能力子服务封装层

`agi.apps` 提供面向具体模态能力的 FastAPI 服务实现，便于独立部署和按需扩缩容。

## 模块划分

- `image/`：文生图与图生图接口。
- `tts/`：文本转语音（TTS）接口。
- `whisper/`：语音转文本（STT）接口。
- `multimodal/`：多模态理解（图文/音频等）接口。
- `embding/`：Embedding 与 Rerank 服务接口。
- `common.py`：鉴权、请求体定义等公共组件。

## 设计特点

- **与主服务解耦**：主服务通过 Base URL 调用能力子服务，而不是强耦合在单进程。
- **模型可替换**：每个子服务可绑定不同模型实现（如 SDXL / SD3.5、CosyVoice、Whisper 等）。
- **部署灵活**：可按 GPU 资源独立部署到不同容器或节点。

## 运行方式（开发）

```bash
make run_image
make run_tts
make run_whisper
make run_hugface
make run_embd
```

## 实践建议

- 将高显存服务（图像、多模态）与轻量服务（embd）分机部署。
- 对外仅暴露主服务端口，子服务置于内网。
- 使用统一的 API Key 与请求追踪 ID，便于审计和排障。
