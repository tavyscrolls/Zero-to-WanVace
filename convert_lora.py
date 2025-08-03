import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import os

def convert_lora_keys(input_lora_path, output_lora_path):
    print(f"Loading LoRA from: {input_lora_path}")
    lora_state_dict = load_file(input_lora_path, device="cpu")

    new_state_dict = {}
    for key, value in lora_state_dict.items():
        new_key = key

        if new_key.startswith("base_model.model."):
            new_key = new_key.replace("base_model.model.", "")

        # The 'diffusion_model.' prefix is standard for the UNet/diffusion part in ComfyUI.
        if not new_key.startswith("diffusion_model."):
             new_key = "diffusion_model." + new_key

        # For original: base_model.model.blocks.0.ffn.0.lora_A.weight
        # -> After step 1: blocks.0.ffn.0.lora_A.weight
        # -> After step 2: diffusion_model.blocks.0.ffn.0.lora_A.weight (Matches ComfyUI)

        print(f"Original LoRA key: {key} -> Converted key: {new_key}")
        new_state_dict[new_key] = value

    print(f"Saving converted LoRA to: {output_lora_path}")
    save_file(new_state_dict, output_lora_path)
    print("Conversion complete!")


if __name__ == "__main__":
    input_lora_path = "adapter_model-1590.safetensors"
    output_lora_path = "adapter_converted-1590.safetensors"

    os.makedirs(Path(output_lora_path).parent, exist_ok=True)

    convert_lora_keys(input_lora_path, output_lora_path)
