from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# 第一次：下载并保存到本地路径
model_name = "google/gemma-3n-E2B-it"
local_dir = "/data/model/gemma-3n-E2B-it"  # 本地保存目录

# 第一次运行时使用此段保存模型
processor = AutoProcessor.from_pretrained(model_name)
model = Gemma3nForConditionalGeneration.from_pretrained(model_name)

# 保存到本地
processor.save_pretrained(local_dir)
model.save_pretrained(local_dir)