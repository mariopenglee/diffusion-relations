from utils import HFDiffusionModel, prompt_preprocessing
import os

def main():
    model_repo_ids = [
        "stabilityai/stable-diffusion-2-1",
        "prompthero/openjourney",
        "CompVis/stable-diffusion-v1-4",
        #"midjourney-community/midjourney-mini",
        #"dalle-mini/dalle-mini",
    ]
    print("Loading models...")
    my_models = [HFDiffusionModel(model_repo_id) for model_repo_id in model_repo_ids]

    with open("data/prompts.txt", "r") as prompts_file:
        prompts = prompts_file.readlines()


    parsed_prompts = prompt_preprocessing(prompts)
    print("Generating images...")
    for prompt in parsed_prompts:
        generated_images = [my_model.generate_image(prompt) for my_model in my_models]
        for i, generated_image in enumerate(generated_images):
            os.makedirs(f"output_images/{i}", exist_ok=True)
            generated_image.images[0].save(f"output_images/{i}/{prompt}.png")
            

if __name__ == "__main__":
    main()
