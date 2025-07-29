from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# 下载并保存 Qwen3-Embedding 模型
embedding_model_path = "/data/model/Qwen3-Embedding-0.6B"
embedding_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
embedding_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

embedding_tokenizer.save_pretrained(embedding_model_path)
embedding_model.save_pretrained(embedding_model_path)

# 下载并保存 Qwen3-Reranker 模型
reranker_model_path = "/data/model/Qwen3-Reranker-0.6B"
reranker_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
reranker_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B")

reranker_tokenizer.save_pretrained(reranker_model_path)
reranker_model.save_pretrained(reranker_model_path)