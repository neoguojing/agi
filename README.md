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
- 知识库支持多租户
- 探索graph将知识库和检索结果实时返回的场景：1.拆分流程；2.流程可以直接返回
