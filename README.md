# personal-ai-image

Fine-tune FLUX 1.dev for personal AI photos. 

## Setup

1. clone toolkit
```bash
conda create -n ai-toolkit python=3.12 -c conda-forge -y && conda activate ai-toolkit
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit && git checkout 5c8fcc8a4efcbfa27fc61609392796c55968f069 && git submodule update --init --recursive && pip install torch && pip install -r requirements.txt
```

2. Make sure you have access to [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 
3. Add `HF_TOKEN` to your environment variables 
```bash
echo "HF_TOKEN=your_token_value" > ai-toolkit/.env
```
4. Make sure your dataset is in the `dataset/` folder with images (`1.jpg`, `2.jpg`, `3.jpg`, etc.) and captions in (`1.txt`, `2.txt`, `3.txt`, etc.)8
5. Modify [`train_lora_flux_personal.yaml`](train_lora_flux_personal.yaml) with your dataset name, trigger word, and other parameters.
6. Run the training script
```bash
python ai-toolkit/run.py train_lora_flux_personal.yaml
```

# Test prompts 

```python
[
    "Philipp, wearing a beanie, sits at a cafe table holding a warm coffee cup.",
    "Philipp builds a chair, surrounded by tools and wood in a workshop.",
    "A smiling Philipp looks directly at the camera for a portrait.",
    "Philipp hikes through a forest trail, carrying a backpack.",
    "Philipp prepares a meal in a kitchen, chopping vegetables.",
    "Philipp plays guitar on a stage, illuminated by spotlights.",
    "Philipp reads a book while relaxing on a park bench.",
    "Philipp paints a canvas with a focused expression.",
    "Philipp rides a bicycle along a scenic coastal road.",
    "Philipp works on a laptop at a desk in a modern office."
]
```


# Tips

Prompt used to caption image with Gemini:

For SD like prompts:
````
Your task is to caption the attached images to train a new text-to-image model the person of the image is me (Philipp), For all captions use [trigger] instead of my name. The caption should be Midjourney, Stable diffusion like.
````
For fine-tuning like prompts:
```bash
Your task is to caption the attached image to train a new text-to-image model. The person in the image is me (Philipp); for all captions, use [trigger] instead of my name. Caption this image in detail, single sentence. Describe what you see.
```