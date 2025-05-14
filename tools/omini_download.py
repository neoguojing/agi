from transformers import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model.save_pretrained("/data/model/Qwen2.5-Omni-3B", from_pt=True)