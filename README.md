# agi
langchain based agi
GPU显存： 初始：2550MB 峰值：20GB
内存： 初始：7GB 峰值：16GB

## RAG
- history_chain = RunnableLambda(self._enter_history, self._aenter_history).with_config(run_name="load_history")
- history_chain = RunnablePassthrough.assign(**{"chat_history": history_chain}).with_config(run_name="insert_history")
- retrieval_docs = (lambda x: x["input"]) | retriever
- retriever = (lambda x: x["input"]) | retriever  or  prompt | llm | StrOutputParser() | retriever, run_name="chat_retriever_chain"
- context=retrieval_docs.with_config(run_name="retrieve_documents")
- "context": format_docs run_name="format_inputs"
- answer =  "context" | prompt | llm | _output_parser     run_name="stuff_documents_chain"

## TODO
- 知识库支持多租户 DONE
- 探索graph将知识库和检索结果实时返回的场景：1.拆分流程；2.流程可以直接返回 DONE
- 支持多模态图片作为输入的ocr解读
- 上传文件之后，其他的问答军基于该文档问答，会导致问题 DONE
- doc文件提取有问题

## OPENWEBUI 修改
- 新增: backend/open_webui/routers/agi.py
- 修改: backend/open_webui/utils/models.py
- 修改: backend/open_webui/utils/middleware.py
- 修改: backend/open_webui/main.py
- 修改: backend/open_webui/config.py
- 修改: backend/open_webui/audio.py
- 修改: .env.example