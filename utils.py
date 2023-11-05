from diffusers import DiffusionPipeline

class HFDiffusionModel:
    def __init__(self, model_repo_id):
        self.model_repo_id = model_repo_id
        self.pipeline = DiffusionPipeline.from_pretrained(model_repo_id, use_safetensors=True)

    def generate_image(self, prompt):
        # Generate an image using the model and prompt
        output_image = self.pipeline(prompt)
        return output_image
    

def prompt_preprocessing(prompts):
    # example prompt: "person|sitting on|chair"
    parsed_prompts = []
    for prompt in prompts:
        prompt = prompt.strip()
        svo = prompt.split("|")
        original_prompt = svo[0] + " " + svo[1] + " " + svo[2]
        reverse_prompt = svo[2] + " " + svo[1] + " " + svo[0]
        parsed_prompts.append(original_prompt)
        parsed_prompts.append(reverse_prompt)
    return parsed_prompts
    


# Usage example
if __name__ == "__main__":
    model_repo_id = "stabilityai/stable-diffusion-2-1"
    my_model = HFDiffusionModel(model_repo_id)
    prompt = "cowboy riding horse"
    generated_image = my_model.generate_image(prompt)
    generated_image.save("output_image.png")

