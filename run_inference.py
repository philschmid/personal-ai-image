import argparse
from diffusers import DiffusionPipeline
import os

def main(lora_id, prompts, num_images_per_prompt):
    pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    pipeline.load_lora_weights(lora_id, weight_name="philipp_flux_1_dev.safetensors")
    pipeline.to("cuda")

    os.makedirs("results", exist_ok=True)
    for prompt in prompts:
      out = pipeline(
                prompt=prompt,
                guidance_scale=4.0,
                num_inference_steps=28,
                num_images_per_prompt=num_images_per_prompt
            )
      
      for i, im in enumerate(out.images):
        im.save(f"results/image_{prompt.replace(' ', '_')[:25]}_{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-id', type=str, default="nielrs/flux-dev-lora-niels", required=True, help='Lora ID to load weights')
    parser.add_argument('--prompts', type=str, nargs='+', default=None, help='List of prompts')
    parser.add_argument('--num-images-per-prompt', type=int, default=4, help='Number of images to generate per prompt')
    
    args = parser.parse_args()
    
    # Convert single string prompt to list
    prompts = args.prompts if isinstance(args.prompts, list) else [args.prompts]
    if prompts is None:
      prompts = [
        "Philipp, wearing a beanie, sits at a cafe table holding a warm coffee cup.",
        "Philipp builds a chair, surrounded by tools and wood in a workshop.",
        "A smiling Philipp looks directly at the camera for a portrait.",
        "Philipp hikes through a forest trail, carrying a backpack.",
        "Philipp prepares a meal in a kitchen, chopping vegetables.",
        "Philipp plays guitar on a stage, illuminated by spotlights.",
        "Philipp reads a book while relaxing on a park bench.",
      ]
    main(args.lora_id, prompts, args.num_images_per_prompt)

# python run_inference.py --lora-id philschmid/flux-test-1 --prompts "A smiling Philipp looks directly at the camera for a portrait." --num-images-per-prompt 5
# srun --nodes=1 --cpus-per-task=12 --gres=gpu:1 --partition=hopper-prod --time 6:50:00 --pty bash