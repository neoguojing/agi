from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
pipe.save_pretrained("/data/model/stable-diffusion-3.5-medium", from_pt=True)