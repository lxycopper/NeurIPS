import os
import argparse

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
from templates.templates import inference_templates

import math

"""
Inference script for generating batch results
"""


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:7"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="input a single text prompt for generation",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        help="select a batch of text prompts from templates.py for generation",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="absolute path to the folder that contains the trained results",
    )
    parser.add_argument(
        "--placeholder_string",
        type=str,
        default="<R>",
        help="place holder string of the relation prompt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="number of samples to generate for each prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="scale for classifier-free guidance",
    )
    parser.add_argument(
        "--inference_folder_name",
        type=str,
        default = "inference-0306-8",
        help="name for the inference_folder",
    )
    args = parser.parse_args()
    return args


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def inference_fn(
        examples: list,
        prompt: str,
        num_samples: int,
        guidance_scale: float,
        ddim_steps: int,
    ) -> Image.Image:

    import pathlib

    """
    same functionality as main(), but for gradio demo usage,
    so slightly modified the input and output format
    """
    # select model_id
    model_id = pathlib.Path(examples[0]).stem

    # create inference pipeline
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(os.path.join('experiments', model_id),torch_dtype=torch.float16).to(args.device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(os.path.join('experiments', model_id)).to('cpu')

    # single text prompt
    if prompt is not None:
        prompt_list = [prompt]
    else:
        prompt_list = []

    for prompt in prompt_list:
        # insert relation prompt <R>
        # prompt = prompt.lower().replace("<r>", "<R>").format(placeholder_string)
        prompt = prompt.lower().replace("<r>", "<R>").format("<R>")

        # batch generation
        images = pipe(prompt, num_inference_steps=ddim_steps, guidance_scale=guidance_scale, num_images_per_prompt=num_samples).images

        # save a grid of images
        image_grid = make_image_grid(images, rows=2, cols=math.ceil(num_samples/2))
        print(image_grid)

        return image_grid


def main():
    args = parse_args()

    # create inference pipeline
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id,torch_dtype=torch.float16).to(args.device)

    # make directory to save images
    inference_folder_name = args.inference_folder_name
    image_root_folder = os.path.join(args.model_id, f'{inference_folder_name}')
    os.makedirs(image_root_folder, exist_ok = True)

    if args.prompt is None and args.template_name is None:
        raise ValueError("please input a single prompt through'--prompt' or select a batch of prompts using '--template_name'.")

    # single text prompt
    if args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = []

    if args.template_name is not None:
        # read the selected text prompts for generation
        prompt_list.extend(inference_templates[args.template_name])
        
    seeds = [1,12,123,1234,12345,123456,1234567,12345678,123456789,12345678910]
    
    for seed in seeds:
        for prompt in prompt_list:
            # insert relation prompt <R>
            prompt = prompt.lower().replace("<r>", "<R>").format(args.placeholder_string)

            # make sub-folder
            image_folder = os.path.join(image_root_folder, prompt, 'samples')
            os.makedirs(image_folder, exist_ok = True)
            # negative_prompts = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner,\
            #     extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting\
            #     bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face,\
            #     three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, \
            #     extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, cartoon, cg, 3d, unreal, animate"
            # batch generation
            images = pipe(prompt, generator=torch.manual_seed(seed),num_inference_steps=50, guidance_scale=args.guidance_scale, num_images_per_prompt=args.num_samples).images

            # save generated images
            for idx, image in enumerate(images):
                image_name = f"seed_{seed}_{str(idx).zfill(4)}.png"
                image_path = os.path.join(image_folder, image_name)
                image.save(image_path)

        # save a grid of images
            image_grid = make_image_grid(images, rows=2, cols=math.ceil(args.num_samples/2))
            image_grid_path = os.path.join(image_root_folder, prompt, f'seed_{seed}.png')
            image_grid.save(image_grid_path)


if __name__ == "__main__":
    main()
