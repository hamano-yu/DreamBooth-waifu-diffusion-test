import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'model/okoma2',
    torch_dtype=torch.float16
).to('cuda')

pipe.enable_attention_slicing() 
pipe.enable_sequential_cpu_offload()
prompt = "okoma"
#prompt = "masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck""masterpiece, best quality, very_high_resolution, large_filesize, full color, 1girl, okoma, outdoors, sky, small breasts"
#prompt = "masterpiece, best quality, 1girl, upper body, outdoors, fantasy, (((kizunanec)))"
#prompt = "masterpiece, best quality, 1girl, upper body, okoma, outdoors"
#prompt = "masterpiece, best quality, very_high_resolution, large_filesize, full color, 1girl, okoma, outdoors, sky, small breasts"
#prompt = "masterpiece, best quality, very_high_resolution, large_filesize, full color, 1girl, outdoors, sky, small breasts"
#prompt = "masterpiece, best quality, 1girl, upper body, okoma, (((fantasy)))"
negative_prompt = "nsfw, owres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,missing fingers,bad hands,missing arms, long neck, Humpbacked"


with autocast("cuda"):
    for i in range(10):
        image = pipe(prompt, guidance_scale=6,height=512,width=512).images[0]
#    image = pipe(prompt, guidance_scale=6,height=768,width=1200).images[0]
     #image = pipe(prompt,negative_prompt = negative_prompt,guidance_scale=6,height=768,width=1024).images[0]
        image.save("images/{:03}.png".format(i))