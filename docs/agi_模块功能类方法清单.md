# AGI 代码模块功能、类与方法清单

> 说明：本清单基于 `agi/` 目录下 Python 源码 AST 自动梳理，覆盖模块级函数、类及类方法（含异步方法）。

## 模块：`agi/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/apps/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/apps/common.py`
- **功能概述**：业务模块实现。
- **模块函数**：
  - `async verify_api_key(credentials)`
- **类与方法**：
  - 类 `SpeechRequest`
    - （无显式方法）
  - 类 `ImageURL`
    - （无显式方法）
  - 类 `MessageContent`
    - （无显式方法）
  - 类 `ChatMessage`
    - （无显式方法）
  - 类 `ChatCompletionRequest`
    - （无显式方法）
  - 类 `ChatResponse`
    - （无显式方法）

## 模块：`agi/apps/embding/embding_model.py`
- **功能概述**：业务模块实现。
- **模块函数**：无。
- **类与方法**：
  - 类 `QwenEmbedding`
    - `def __init__(self, model_path, timeout)`
    - `def get_model(self)`
    - `def _load(self)`
    - `def last_token_pool(last_hidden_states, attention_mask)`
    - `def embed_query(self, query, dimension)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/apps/embding/fast_api_embding.py`
- **功能概述**：API 接口封装。
- **模块函数**：
  - `async embed(request)`
  - `async rerank_api(request)`
- **类与方法**：
  - 类 `OllamaEmbedRequest`
    - （无显式方法）
  - 类 `OllamaEmbedResponse`
    - （无显式方法）
  - 类 `RerankItem`
    - （无显式方法）
  - 类 `RerankResponse`
    - （无显式方法）
  - 类 `RerankRequest`
    - （无显式方法）

## 模块：`agi/apps/embding/rerank.py`
- **功能概述**：重排模型封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `Reranker`
    - `def __init__(self, timeout, device, max_length)`
    - `def get_model(self, model)`
    - `def _load(self)`
    - `def format_instruction(self, instruction, query, doc)`
    - `def process_inputs(self, pairs)`
    - `def compute_logits(self, inputs)`
    - `def rerank(self, queries, documents, model, instruction)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/apps/image/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/apps/image/fast_api_image.py`
- **功能概述**：API 接口封装。
- **模块函数**：
  - `async generate(req, api_key)`
  - `async chat_completion(request, api_key)`
- **类与方法**：
  - 类 `ImageGenRequest`
    - （无显式方法）
  - 类 `UrlData`
    - （无显式方法）
  - 类 `B64Data`
    - （无显式方法）
  - 类 `ImageGenResponse`
    - （无显式方法）

## 模块：`agi/apps/image/image2image.py`
- **功能概述**：图像处理/生成任务。
- **模块函数**：无。
- **类与方法**：
  - 类 `Image2Image`
    - `def __init__(self, model_path, timeout, save_image)`
    - `def get_model(self)`
    - `def _load(self)`
    - `def invoke(self, input, input_image, resp_format)`
    - `def handle_output(self, image, html)`
    - `def _save_or_resize_image(self, image)`
    - `def _save_image(self, image)`
    - `def _convert_image_to_base64(self, image)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/apps/image/text2image.py`
- **功能概述**：图像处理/生成任务。
- **模块函数**：无。
- **类与方法**：
  - 类 `Text2Image`
    - `def __init__(self, model_path, timeout, save_image)`
    - `def get_model(self, model)`
    - `def _load(self)`
    - `def invoke(self, input, negative_prompt, model, width, height, resp_format, randomize_seed)`
    - `def handle_output(self, image, html)`
    - `def _save_or_resize_image(self, image)`
    - `def _save_image(self, image)`
    - `def _convert_image_to_base64(self, image)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/apps/multimodal/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/apps/multimodal/fast_api_multimodal.py`
- **功能概述**：API 接口封装。
- **模块函数**：
  - `async chat_completion(request, api_key)`
- **类与方法**：无。

## 模块：`agi/apps/multimodal/multi_modal.py`
- **功能概述**：业务模块实现。
- **模块函数**：无。
- **类与方法**：
  - 类 `MultiModel`
    - `def __init__(self, model_path, timeout, save_file)`
    - `def get_model(self, model)`
    - `def _load(self)`
    - `def invoke(self, model, text, audio, image, video, return_audio, return_fmt)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/apps/tts/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/apps/tts/fast_api_audio.py`
- **功能概述**：API 接口封装。
- **模块函数**：
  - `async audio_stream_ws(websocket, tenant_id)`
  - `async generate_speech(request, api_key)`
  - `async generate_speech_streaming(request, api_key)`
  - `def audio_generator(tenant_id)`
- **类与方法**：无。

## 模块：`agi/apps/tts/tts.py`
- **功能概述**：文本转语音封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `TTS`
    - `def __init__(self, model_path, timeout)`
    - `def get_queue(cls, tenant_id)`
    - `def get_model(self, model_name)`
    - `def _load(self)`
    - `def invoke(self, input_str, user_id, save_file, model_name)`
    - `def generate_audio_samples(self, text)`
    - `def save_audio_to_file(self, text, file_path)`
    - `def uniform_model_output(self, obj)`
    - `def np_pcm_to_wave(self, audio_array, sample_rate, to_base64)`
    - `def list_pcm_normalization_int16(self, audio_data)`
    - `def is_chinese_text(self, text)`
    - `def sentence_segmenter(self, text, min_length, max_length)`
    - `def send_pcm(self, tenant_id, pcm_np, chunk_size)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/apps/utils.py`
- **功能概述**：通用工具函数。
- **模块函数**：
  - `def pick_free_device(threshold_ratio)`
  - `def best_torch_dtype()`
- **类与方法**：无。

## 模块：`agi/apps/whisper/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/apps/whisper/fast_api_whisper.py`
- **功能概述**：API 接口封装。
- **模块函数**：
  - `async transcribe_audio(file, model, prompt, language, response_format, temperature, api_key)`
  - `def compress_audio(file_path)`
- **类与方法**：无。

## 模块：`agi/apps/whisper/speech2text.py`
- **功能概述**：语音转文本封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `Speech2Text`
    - `def __init__(self, model_path, timeout, compute_type)`
    - `def get_model(self, device)`
    - `def _load(self)`
    - `def invoke(self, audio_input, device)`
    - `def _unload(self)`
    - `def _monitor(self)`

## 模块：`agi/config.py`
- **功能概述**：配置管理与环境变量读取。
- **模块函数**：
  - `def get_env_bool(name, default)`
  - `def init_langchain_debug()`
  - `def init_logger()`
- **类与方法**：无。

## 模块：`agi/download.py`
- **功能概述**：模型与资源下载入口。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/fast_api_file.py`
- **功能概述**：API 接口封装。
- **模块函数**：
  - `async list_files()`
  - `async save_file(file, user_id, collection_name)`
  - `async download_file(file_name, request)`
  - `async delete_file(file_name)`
- **类与方法**：无。

## 模块：`agi/fastapi_agi.py`
- **功能概述**：FastAPI 服务入口与路由。
- **模块函数**：
  - `async chat_completions(request, api_key)`
  - `def handle_response_content_as_string(content)`
  - `def format_non_stream_response(resp, web)`
  - `async generate_stream_response(state_data, web)`
  - `async list_models(api_key)`
- **类与方法**：
  - 类 `Model`
    - （无显式方法）
  - 类 `ModelListResponse`
    - （无显式方法）

## 模块：`agi/llms/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/llms/base.py`
- **功能概述**：业务模块实现。
- **模块函数**：
  - `def parse_input_messages(input)`
- **类与方法**：
  - 类 `CustomerLLM`
    - `def __init__(self, **kwargs)`
    - `def destroy(self)`
    - `def encode(self, input)`
    - `def decode(self, ids)`
    - `def model_name(self)`

## 模块：`agi/llms/image2image.py`
- **功能概述**：图像处理/生成任务。
- **模块函数**：无。
- **类与方法**：
  - 类 `Image2Image`
    - `def __init__(self, **kwargs)`
    - `def _llm_type(self)`
    - `def model_name(self)`
    - `def invoke(self, input, config, **kwargs)`

## 模块：`agi/llms/model_factory.py`
- **功能概述**：工厂模式创建对象。
- **模块函数**：无。
- **类与方法**：
  - 类 `ModelFactory`
    - `def get_model(model_type)`
    - `def _load_model(model_type)`

## 模块：`agi/llms/multimodel.py`
- **功能概述**：多模态模型封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `MultiModel`
    - `def __init__(self, **kwargs)`
    - `def model_name(self)`
    - `def invoke(self, inputs, config, **kwargs)`

## 模块：`agi/llms/rerank.py`
- **功能概述**：重排模型封装。
- **模块函数**：
  - `async rerank_batch(client, endpoint, query, documents, model, top_k)`
  - `async safe_rerank_batch(*args, **kwargs)`
  - `async rerank_with_batching(query, documents, endpoint, model, top_k, batch_size)`
- **类与方法**：
  - 类 `RerankItem`
    - `def __init__(self, object, index, document, score)`
    - `def to_dict(self)`

## 模块：`agi/llms/speech2text.py`
- **功能概述**：语音转文本封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `Speech2Text`
    - `def __init__(self, **kwargs)`
    - `def model_name(self)`
    - `def invoke(self, input, config, **kwargs)`

## 模块：`agi/llms/text2image.py`
- **功能概述**：图像处理/生成任务。
- **模块函数**：无。
- **类与方法**：
  - 类 `Text2Image`
    - `def __init__(self, **kwargs)`
    - `def _llm_type(self)`
    - `def model_name(self)`
    - `def invoke(self, input, config, **kwargs)`

## 模块：`agi/llms/tts.py`
- **功能概述**：文本转语音封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `TextToSpeech`
    - `def __init__(self, **kwargs)`
    - `def model_name(self)`
    - `def invoke(self, input, config, **kwargs)`
    - `def stream(self, input, config, **kwargs)`

## 模块：`agi/tasks/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/tasks/agent.py`
- **功能概述**：智能体任务编排。
- **模块函数**：
  - `def _get_state_value(state, key, default)`
  - `def _get_prompt_runnable(prompt)`
  - `def _convert_modifier_to_prompt(func)`
  - `def _should_bind_tools(model, tools)`
  - `def _get_model(model)`
  - `def _validate_chat_history(messages)`
  - `def human_feedback_node(state, config)`
  - `async ahuman_feedback_node(state, config)`
  - `def create_react_agent(model, tools, prompt, response_format, pre_model_hook, state_schema, config_schema, checkpointer, store, interrupt_before, interrupt_after, debug, version, name)`
  - `def modify_state_messages(state)`
  - `def pre_model_hook(state)`
  - `def create_react_agent_task(llm)`
  - `def create_react_agent_as_subgraph(llm)`
- **类与方法**：
  - 类 `AgentStatePydantic`
    - （无显式方法）
  - 类 `AgentStateWithStructuredResponse`
    - （无显式方法）
  - 类 `AgentStateWithStructuredResponsePydantic`
    - （无显式方法）

## 模块：`agi/tasks/agi_prompt.py`
- **功能概述**：提示词模板与构造。
- **模块函数**：
  - `def _create_template_from_message_type(message_type, template, template_format)`
  - `def _convert_to_message(message, template_format)`
- **类与方法**：
  - 类 `_TextTemplateParam`
    - （无显式方法）
  - 类 `_ImageTemplateParam`
    - （无显式方法）
  - 类 `_AudioTemplateParam`
    - （无显式方法）
  - 类 `_VideoTemplateParam`
    - （无显式方法）
  - 类 `MultiModalMessagePromptTemplate`
    - `def get_lc_namespace(cls)`
    - `def from_template(cls, template, template_format, partial_variables, **kwargs)`
    - `def from_template_file(cls, template_file, input_variables, **kwargs)`
    - `def format_messages(self, **kwargs)`
    - `async aformat_messages(self, **kwargs)`
    - `def input_variables(self)`
    - `def format(self, **kwargs)`
    - `async aformat(self, **kwargs)`
    - `def pretty_repr(self, html)`
  - 类 `MultiModalChatPromptTemplate`
    - `def __init__(self, messages, template_format, **kwargs)`

## 模块：`agi/tasks/audio.py`
- **功能概述**：音频处理任务。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/tasks/chat_with_hisrory.py`
- **功能概述**：业务模块实现。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/tasks/cluster.py`
- **功能概述**：业务模块实现。
- **模块函数**：
  - `def get_hdbscan_params(embeddings)`
  - `def get_dpmeans_params(embeddings)`
  - `def train(collection_name, docs, embeddings, cluster_algo)`
- **类与方法**：
  - 类 `TextClusterer`
    - `def __init__(self, collection_name, cluster_algo, hnsw_m, ef_search, distance_threshold, candidate_k, min_cluster_size, min_samples, use_umap, umap_dim, umap_n_neighbors, umap_min_dist)`
    - `def summary(self, text)`
    - `def cluster_range_reward(self, n_clusters, n_samples, target_min, target_max)`
    - `def evaluate_clusters(self, embeddings, labels, sample_size, random_state, cluster_penalty)`
    - `def combined_keywords(self, all_keywords)`
    - `def _reduce_dim(self, vectors)`
    - `def do_hdbscan(self, embeddings)`
    - `def do_dpmeans(self, embeddings)`
    - `def cluster(self, embeddings)`
    - `def post_processor(self, docs, labels)`

## 模块：`agi/tasks/db_builder.py`
- **功能概述**：向量库/知识库构建。
- **模块函数**：
  - `async file_loader_node(state, config)`
  - `async doc_split_node(state, config)`
  - `async doc_clean_node(state, config)`
  - `async doc_embding_node(state, config)`
  - `async doc_keywords_node(state, config)`
  - `async cluster_train_node(state, config)`
  - `async store_index_node(state, config)`
  - `async store_node(state, config)`
  - `async last_node(state, config)`
- **类与方法**：无。

## 模块：`agi/tasks/define.py`
- **功能概述**：业务模块实现。
- **模块函数**：无。
- **类与方法**：
  - 类 `Feature`
    - （无显式方法）
  - 类 `InputType`
    - （无显式方法）
  - 类 `AgentState`
    - （无显式方法）
  - 类 `State`
    - （无显式方法）
  - 类 `AskHuman`
    - （无显式方法）

## 模块：`agi/tasks/file_loader.py`
- **功能概述**：文件加载与切分。
- **模块函数**：
  - `def get_file_loader(file_path, file_content_type)`
  - `def resolve_hostname(hostname)`
  - `def validate_url(url)`
  - `def get_web_loader(url, verify_ssl)`
  - `def get_youtube_loader(url)`
- **类与方法**：
  - 类 `InvalidURLException`
    - （无显式方法）

## 模块：`agi/tasks/graph.py`
- **功能概述**：图工作流编排。
- **模块函数**：无。
- **类与方法**：
  - 类 `AgiGraph`
    - `def __init__(self)`
    - `async routes(self, state, config)`
    - `async auto_state_machine(self, state)`
    - `async image_feature_control(self, state)`
    - `async audio_feature_control(self, state)`
    - `async video_feature_control(self, state)`
    - `async text_feature_control(self, state)`
    - `async human_feedback_control(self, state)`
    - `async tts_prepare_node(self, state, config)`
    - `async web_search_node(self, state, config)`
    - `async rag_search_node(self, state, config)`
    - `async web_scrape_node(self, state, config)`
    - `async output_control(self, state)`
    - `async invoke(self, input)`
    - `async stream(self, input, stream_mode)`
    - `def display(self)`

## 模块：`agi/tasks/image.py`
- **功能概述**：图像处理/生成任务。
- **模块函数**：
  - `async intend_understand_modify_state_messages(state)`
  - `async intend_understand_node(state, config)`
- **类与方法**：无。

## 模块：`agi/tasks/llm.py`
- **功能概述**：LLM 相关任务编排。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/tasks/llm_app.py`
- **功能概述**：LLM 相关任务编排。
- **模块函数**：
  - `def is_valid_url(url)`
  - `def get_session_history(user_id, conversation_id)`
  - `def create_llm_with_history(runnable, dict_input)`
  - `def create_stuff_documents_chain(llm, prompt, output_parser, document_prompt, document_separator)`
  - `def create_websearch(km)`
  - `def create_rag(km)`
  - `def create_chat(llm)`
  - `def create_chat_with_history(llm)`
  - `def build_citations(documents)`
- **类与方法**：无。

## 模块：`agi/tasks/multi_model_app.py`
- **功能概述**：业务模块实现。
- **模块函数**：
  - `def create_translate_chain(llm)`
  - `def multimodel_state_modifier(state)`
  - `def create_text2image_chain(llm)`
  - `def create_image_gen_chain(llm)`
  - `def create_text2speech_chain()`
  - `def create_speech2text_chain()`
  - `def create_llm_task(**kwargs)`
  - `def create_embedding_task(**kwargs)`
  - `def create_multimodel_chain()`
- **类与方法**：无。

## 模块：`agi/tasks/multimodel.py`
- **功能概述**：多模态模型封装。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/tasks/prompt.py`
- **功能概述**：提示词模板与构造。
- **模块函数**：
  - `def traslate_modify_state_messages(state)`
  - `def decide_modify_state_messages(state)`
  - `def default_modify_state_messages(state)`
  - `def docqa_modify_state_messages(state)`
  - `def tts_modify_state_messages(state)`
- **类与方法**：
  - 类 `YesNoOutputParser`
    - `def parse(self, text)`

## 模块：`agi/tasks/rag_web.py`
- **功能概述**：RAG 检索增强生成。
- **模块函数**：
  - `async intend_understand_modify_state_messages(state)`
  - `def get_clusterid_collection_pair(docs)`
  - `def refine_query(feature, query)`
  - `async doc_chat_node(state, config, writer)`
  - `async doc_rerank_node(state, config)`
  - `async doc_summary_node(state, config)`
  - `async web_search_node(state, config)`
  - `async web_scrape_node(state, config)`
  - `async doc_split_node(state, config)`
  - `async context_control(state)`
  - `async rag_auto_route(state)`
  - `async route(state)`
  - `async index_search_node(state, config)`
  - `async search_node(state, config)`
- **类与方法**：无。

## 模块：`agi/tasks/retriever.py`
- **功能概述**：检索器与向量检索逻辑。
- **模块函数**：无。
- **类与方法**：
  - 类 `SourceType`
    - （无显式方法）
  - 类 `FilterType`
    - （无显式方法）
  - 类 `SimAlgoType`
    - （无显式方法）
  - 类 `KnowledgeManager`
    - `def __init__(self, data_path, llm, embedding)`
    - `def list_documets(self, collection_name, tenant)`
    - `def list_collections(self, tenant)`
    - `async store(self, collection_name, source, tenant, source_type, **kwargs)`
    - `def get_compress_retriever(self, filter_type)`
    - `def bm25_retriever(self, docs, k)`
    - `def get_retriever(self, collection_names, tenant, k, bm25, filter_type, sim_algo)`
    - `async query_doc(self, collection_name, query, tenant, k, bm25, filter_type, to_dict)`
    - `def resolve_hostname(self, hostname)`
    - `def validate_url(self, url)`
    - `def get_web_loader(self, url, verify_ssl)`
    - `def get_youtube_loader(self, url)`
    - `def get_file_loader(self, filename, file_path, file_content_type)`
    - `async web_parser(self, urls, tenant, metadata, collection_name)`
    - `async web_search(self, query, max_results, bm25)`
    - `async split_documents(self, documents, chunk_size, chunk_overlap)`
    - `def do_search(self, questions)`
    - `async do_asearch(self, questions)`

## 模块：`agi/tasks/task_factory.py`
- **功能概述**：工厂模式创建对象。
- **模块函数**：
  - `def create_llm_chat_task(**kwargs)`
  - `def create_llm_with_history_task(**kwargs)`
  - `def create_rag_task(**kwargs)`
  - `def create_web_search_task(**kwargs)`
  - `def create_docchain_task(**kwargs)`
  - `def create_translate_task(**kwargs)`
  - `def create_image_gen_task(**kwargs)`
  - `def create_tts_task(**kwargs)`
  - `def create_speech_text_task(**kwargs)`
  - `def create_multimodel_task(**kwargs)`
  - `def create_agent_task(**kwargs)`
- **类与方法**：
  - 类 `TaskFactory`
    - `def get_knowledge_manager()`
    - `def get_embedding(model)`
    - `def get_llm()`
    - `def get_thinking_llm()`
    - `def get_llm_with_output_format(debug)`
    - `def create_task(task_type, **kwargs)`

## 模块：`agi/tasks/tools.py`
- **功能概述**：业务模块实现。
- **模块函数**：
  - `def wikipedia()`
  - `def wikidata()`
  - `def pythonREPL()`
  - `async search(query)`
  - `async web_scrape(query)`
- **类与方法**：无。

## 模块：`agi/tasks/utils.py`
- **功能概述**：通用工具函数。
- **模块函数**：
  - `def split_think_content(content)`
  - `def get_text_from_message(message)`
  - `def get_last_message_text(state)`
  - `def refine_human_message(state, formatter)`
  - `def refine_last_message_text(message)`
  - `def graph_response_format(message)`
  - `def format_state_message_to_str(messages)`
  - `def debug_info(x)`
  - `def compute_content_hash(content)`
  - `def add_messages(left, right)`
  - `def save_media_content(source, output_dir)`
  - `def image_path_to_base64_uri(image_path)`
  - `def graph_print(graph)`
  - `def audio_to_base64(audio_path)`
- **类与方法**：无。

## 模块：`agi/tasks/vectore_store.py`
- **功能概述**：业务模块实现。
- **模块函数**：无。
- **类与方法**：
  - 类 `CollectionManager`
    - `def __init__(self, data_path, embedding, allow_reset, anonymized_telemetry)`
    - `def get_or_create_tenant_for_user(self, tenant, database)`
    - `def _get_persistent_client(self, tenant, database)`
    - `def client(self, tenant, database)`
    - `def get_or_create_collection(self, collection_name, tenant, database)`
    - `def delete_collection(self, collection_name, tenant, database)`
    - `def list_collections(self, limit, offset, tenant, database)`
    - `def get_vector_store(self, collection_name, tenant, database)`
    - `def get_documents(self, collection_name, source, limit, offset, tenant, database)`
    - `def get_sources(self, collection_name, tenant, database)`
    - `async add_documents(self, documents, collection_name, embeddings, ids, batch_size, tenant, database)`
    - `async embedding_search(self, texts, collection_name, k, cluster_id, tenant, database)`
    - `async full_search(self, texts, collection_name, cluster_id, k, tenant, database)`
    - `def build_query(self, contains_list, not_contains_list)`

## 模块：`agi/tasks/video.py`
- **功能概述**：视频处理任务。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/utils/__init__.py`
- **功能概述**：包初始化与对外导出定义。
- **模块函数**：无。
- **类与方法**：无。

## 模块：`agi/utils/common.py`
- **功能概述**：业务模块实现。
- **模块函数**：
  - `def remove_data_uri_prefix(data_uri)`
  - `def is_url(input_str)`
  - `def is_base64(input_str)`
  - `def download_file(url, target_path)`
  - `def save_base64_to_file(b64_data, target_path)`
  - `def guess_extension(mime_type)`
  - `def guess_type(filepath)`
  - `def classify_mime(mime_type)`
  - `def detect_input_and_save(input_data, target_path)`
  - `def file_to_data_uri(file_path)`
  - `def is_relative_path(path_str)`
  - `def identify_input_type(input_str)`
  - `def path_to_preview_url(file_path, base_url)`
- **类与方法**：
  - 类 `Timer`
    - `def __enter__(self)`
    - `def __exit__(self, *args)`
  - 类 `Media`
    - `def from_data(cls, input_data, media_type)`
    - `def load_image(image)`
    - `def load_audio(audio)`
    - `def to_binary(self)`
    - `def save_to_file(self, file_path)`

## 模块：`agi/utils/file_storage.py`
- **功能概述**：RAG 检索增强生成。
- **模块函数**：
  - `async compute_sha256(file, chunk_size)`
- **类与方法**：
  - 类 `StorageError`
    - （无显式方法）
  - 类 `FileNotFound`
    - （无显式方法）
  - 类 `InvalidFilename`
    - （无显式方法）
  - 类 `FSSpecStorage`
    - `def __init__(self, base_path)`
    - `def _full_path(self, filename)`
    - `def exists(self, filename)`
    - `def to_local_path(self, filename)`
    - `def path_to_preview_url(self, filename, base_url)`
    - `async save(self, file_obj, filename)`
    - `async load(self, filename)`
    - `async delete(self, filename)`
    - `async list_files(self)`
    - `def get_mime_type(self, filename)`
  - 类 `FileService`
    - `def __init__(self, storage)`
    - `def generate_unique_filename(self, original_name)`
    - `async generate_hashed_filename(self, file, original_name)`
    - `async save_file(self, file, original_name)`

## 模块：`agi/utils/nlp.py`
- **功能概述**：自然语言处理。
- **模块函数**：无。
- **类与方法**：
  - 类 `TextProcessor`
    - `def __init__(self, stop_words_path, user_dict_path, top_k, filtered_flags)`
    - `def detect_language(self, text)`
    - `def tokenize(self, text)`
    - `def tokenize_with_pos(self, text)`
    - `def load_stopwords(self, stop_words_path)`
    - `def remove_stopwords(self, text)`
    - `def remove_stopwords_batch(self, texts)`
    - `def _clean_text(self, text)`
    - `def clean_batch(self, texts)`
    - `def extract_keywords(self, text, method)`
    - `def batch_process(self, texts, method)`
    - `async abatch_process(self, texts, method)`

## 模块：`agi/utils/scrape.py`
- **功能概述**：网页抓取。
- **模块函数**：无。
- **类与方法**：
  - 类 `WLInput`
    - （无显式方法）
  - 类 `WebScraper`
    - `def __init__(self, web_paths, **kwargs)`
    - `def load(self)`
    - `def _run(self, web_paths, run_manager)`
    - `async aload(self, web_paths)`
    - `async aload2(self, question_url_map)`
    - `async _fetch_and_parse(self, url)`
    - `async _fetch(self, url)`
    - `async _fetch_playwright(self, url)`
    - `def _random_ua(self)`
    - `def _parse_local(self, html, source)`
    - `def _is_noise(self, line)`
    - `def _safe_text(self, tag)`

## 模块：`agi/utils/search_engine.py`
- **功能概述**：搜索引擎封装。
- **模块函数**：无。
- **类与方法**：
  - 类 `SGInput`
    - （无显式方法）
  - 类 `SearchEngineSelector`
    - `def __init__(self, **kwargs)`
    - `def record_result(self, engine_name, success)`
    - `def get_success_rate(self, engine_name)`
    - `def select_engine(self)`
    - `def get_engine(self, name)`
    - `def _run(self, query, run_manager)`
    - `async batch_search(self, questions)`

## 模块：`agi/utils/stock_market.py`
- **功能概述**：股票市场工具。
- **模块函数**：
  - `def get_stock(input, topk)`
- **类与方法**：
  - 类 `StockData`
    - （无显式方法）

## 模块：`agi/utils/tika.py`
- **功能概述**：Tika 文档解析。
- **模块函数**：无。
- **类与方法**：
  - 类 `TikaExtractor`
    - `def __init__(self, file_path, tika_url)`
    - `def lazy_load(self)`
    - `def extract_text(self, file_path, output, accept, html_to_text)`
    - `def extract_metadata(self, file_path, accept)`
    - `def _guess_content_type(self, file_path)`

## 模块：`agi/utils/weather.py`
- **功能概述**：天气工具。
- **模块函数**：
  - `def get_province_list()`
  - `def get_city_list(province_code)`
  - `def build_search_list()`
  - `def find_best_match(input_name)`
  - `def get_nmc_weather(station_id, timeout)`
  - `def get_weather_info(city_name)`
- **类与方法**：
  - 类 `WeatherAPIError`
    - （无显式方法）

## 模块：`agi/utils/yacy.py`
- **功能概述**：YaCy 搜索接入。
- **模块函数**：无。
- **类与方法**：
  - 类 `YaCySearch`
    - `def __init__(self, max_results)`
    - `async start_yacy_crawl_async(self, target_url, username, password)`
    - `async yacy_search_async(self, query, start_record, maximum_records, contentdom, resource, urlmaskfilter, prefermaskfilter, verify, lr, meancount, nav)`
    - `def search(self, query, start_record, maximum_records, contentdom, resource, urlmaskfilter, prefermaskfilter, verify, lr, meancount, nav)`
