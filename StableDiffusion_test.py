from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")  # Move to GPU if available

# Generate an image
prompt = "A painting of a cat playing a guitar"
image = pipe(prompt).images[0]

# Save the image
image.save("cat_guitar.png")