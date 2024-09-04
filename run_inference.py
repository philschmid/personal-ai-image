import argparse
from diffusers import DiffusionPipeline
import os
import torch

def main(lora_id, prompts, num_images_per_prompt):
    pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipeline.load_lora_weights("/fsx/philipp/personal-ai-image/output/personal_flux_1_dev", weight_name="personal_flux_1_dev.safetensors")
    
    pipeline.to("cuda")

    os.makedirs("results", exist_ok=True)
    for prompt in prompts:
      image_dir = f"results/{prompt.replace(' ', '_')[:100]}"
      os.makedirs(image_dir, exist_ok=True)
      # write prompt to file
      with open(f"{image_dir}/prompt.txt", "w") as f:
        f.write(prompt)
      for id in range(num_images_per_prompt):
        out = pipeline(
                  prompt=prompt,
                  guidance_scale=3.5,
                  num_inference_steps=20,
                  num_images_per_prompt=4
              )
        
        for i, im in enumerate(out.images):
          im.save(f"{image_dir}/{prompt.replace(' ', '_')[:75]}_{id}_{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-id', type=str, default="nielrs/flux-dev-lora-niels", required=True, help='Lora ID to load weights')
    # parser.add_argument('--prompts', type=str, nargs='+', default=None, help='List of prompts')
    parser.add_argument('--num-images-per-prompt', type=int, default=4, help='Number of images to generate per prompt')
    
    args = parser.parse_args()
    
    prompts = [
  "A portrait of Philipp",
  "A portrait of Philipp with a beard",
  "A portrait of Philipp with a buzz cut",
  "A portrait of Philipp with a slight smile",
  "A portrait of Philipp with a buzz cut and a short beard, slightly smiling",
  "A portrait of Philipp with a buzz cut and a short beard, slightly smiling",
  "Picture of Philipp in Tokyo at night, shot with a wide-angle lens (24mm) at f/1.8. Use a shallow depth of field to focus on Philipp, with the glowing street signs and bustling crowd blurred in the background. High ISO setting to capture the ambient light, giving the image a slight grain for a cinematic feel."
  "A portait of Philipp, with a short beard, buzz cut, natural with no makeup, the skin pores and texture are visible, the face is slightly tanned, the eyes are smiling"
  ]
    
    
    # Convert single string prompt to list
    # prompts = args.prompts if isinstance(args.prompts, list) else [args.prompts]
    # if prompts is None:
    # prompts = [
    #     "Philipp, wearing a beanie, sits at a cafe table holding a warm coffee cup.",
    #     "Philipp builds a chair, surrounded by tools and wood in a workshop.",
    #     "A smiling Philipp looks directly at the camera for a portrait.",
    #     "Philipp hikes through a forest trail, carrying a backpack.",
    #     "Philipp prepares a meal in a kitchen, chopping vegetables.",
    #     "Philipp plays guitar on a stage, illuminated by spotlights.",
    #     "Philipp reads a book while relaxing on a park bench.",
    #   ]
    
   
    
    main(args.lora_id, prompts, args.num_images_per_prompt)

# python run_inference.py --lora-id philschmid/flux-test-1 --prompts "A smiling Philipp with a buzz cut looks directly at the camera for a portrait." --num-images-per-prompt 4
# srun --nodes=1 --cpus-per-task=12 --gres=gpu:1 --partition=hopper-prod --time 6:50:00 --pty bash


