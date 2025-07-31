from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
reranker_model_path = "/data/model/bge-reranker-v2-m3"
tokenizer.save_pretrained(reranker_model_path)
model.save_pretrained(reranker_model_path)