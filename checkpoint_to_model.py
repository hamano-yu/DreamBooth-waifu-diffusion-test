from accelerate import Accelerator
from diffusers import DiffusionPipeline

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "hakurei/waifu-diffusion"
#model_id = "model/okoma2"
pipeline = DiffusionPipeline.from_pretrained(model_id)
accelerator = Accelerator()

#Use text_encoder if `--train_text_encoder` was used for the initial training
#unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

# Restore state from a checkpoint path. You have to use the absolute path here.
print("load checkpoint")
model_path = "~/Desktop/projects/stable_diffusion/waifu-diffusion/model/okoma/checkpoint-200"
accelerator.load_state(model_path)

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
#    unet=accelerator.unwrap_model(unet),
#    text_encoder=accelerator.unwrap_model(text_encoder),
)

# Perform inference, or save, or push to the hub
pipeline.save_pretrained("model-checkpoint")