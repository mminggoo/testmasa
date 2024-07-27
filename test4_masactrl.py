import abc
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers.configuration_utils import FrozenDict, deprecate
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models.attention import Attention
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl

from config import RunConfig

args = RunConfig()

def transform_variable_name(input_str, attn_map_save_step):
    # Split the input string into parts using the dot as a separator
    parts = input_str.split('.')

    # Extract numerical indices from the parts
    # indices = [int(part) if part.isdigit() else part for part in parts]
    indices = [f"[{part}]" if part.isdigit() else f".{part}" for part in parts]
    indices = "".join(indices)

    # Build the desired output string
    # output_str = f'pipe.unet.{indices[0]}[{indices[1]}].{indices[2]}[{indices[3]}].{indices[4]}[{indices[5]}].{indices[6]}.attn_map[{attn_map_save_step}]'
    output_str = f'pipe.unet{indices}.attn_map[{attn_map_save_step}]'.replace('.processor', '')

    return output_str

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def find_bounding_box(mask_image_path):
    # 마스크 이미지 읽기
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (64, 64))
    
    # 이진화 (Thresholding)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 외곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 외곽선 찾기
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 바운딩 박스 계산
    x, y, w, h = cv2.boundingRect(largest_contour)

    height, width = mask.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    # 바운딩 박스 안쪽을 흰색으로 설정
    mask[y:y+h, x:x+w] = 255


    mask2 = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.resize(mask2, (512, 512))
    _, binary = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return mask, x, y, w, h


LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pipe = MasaCtrlPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None).to(device)
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None).to(device)
# pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', dtype=torch.float16, safety_checker=None).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
tokenizer = pipe.tokenizer


from PIL import Image
from tqdm import tqdm
from torchvision.io import read_image
import yaml, os

source_images = []
generated_images = []
eval_prompts = []


output_dir_root = args.output_dir_root

inverse_prompt = ''

SOURCE_IMAGE_PATH = args.source

target_prompt = args.prompt
targets = args.targets

print(targets)

ids = pipe.tokenizer(target_prompt).input_ids
indices = {i: tok for tok, i in zip(pipe.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
print(indices)

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

source_image = load_image(SOURCE_IMAGE_PATH, device)

latents, latents_list = pipe.invert(source_image,
                                    inverse_prompt,
                                    guidance_scale=7.5,
                                    num_inference_steps=50,
                                    return_intermediates=True,
                                    )

prompts = ['', # args[key]['origin_prompt']
            target_prompt] # target_prompt

latents = latents.expand((2, 4, 64, 64))


mask, x, y, w, h = find_bounding_box(f'OIR-Bench/new_multi_object/mask/{SOURCE_IMAGE_PATH.split("/")[-1].split(".")[0]}/{args[key]["origin_change"][0]}.png')
mask = torch.from_numpy(mask).float().to(device) # multi_object_001/a cat.png multi_object_002/leaves.png multi_object_003/cat on the left.png
mask = torch.where(mask > 0, 1.0, 0.0).float().to(device).view(1, 1, 64, 64)

mask2, x2, y2, w2, h2 = find_bounding_box(f'OIR-Bench/new_multi_object/mask/{SOURCE_IMAGE_PATH.split("/")[-1].split(".")[0]}/{args[key]["origin_change"][1]}.png')
mask2 = torch.from_numpy(mask2).float().to(device) # multi_object_001/a cat.png multi_object_002/leaves.png multi_object_003/cat on the left.png
mask2 = torch.where(mask2 > 0, 1.0, 0.0).float().to(device).view(1, 1, 64, 64)

masks = [mask, mask2]

# mask = torch.from_numpy(np.array(Image.open(f'OIR-Bench/multi_object/mask/{SOURCE_IMAGE_PATH.split("/")[-1].split(".")[0]}/{args[key]["origin_change"][0]}.png').resize((64, 64)))).float().to(device) # multi_object_001/a cat.png multi_object_002/leaves.png multi_object_003/cat on the left.png
# mask = torch.where(mask > 0, 1.0, 0.0)
# mask = cv2.dilate(mask.cpu().numpy(), np.ones((3, 3), np.uint8), iterations=10)
# mask = torch.from_numpy(mask).float().to(device).view(1, 1, 64, 64)


source = np.array(Image.open(SOURCE_IMAGE_PATH).resize((512, 512)))
cv2.rectangle(source, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.rectangle(source, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
new_image = Image.new('RGB', (512*6, 512))
new_image.paste(Image.fromarray(source), (0, 0))

for idx_num, grad_scale in enumerate([50, 500, 1000, 2500, 5000]):
    editor = MutualSelfAttentionControl(8, 10)
    regiter_attention_editor_diffusers(pipe, editor)
    pipe.controller = editor

    outputs = pipe(prompt=prompts, height=512, width=512, num_inference_steps=50, latents=latents.clone(),
                ref_intermediate_latents=latents_list,
                masks=masks,
                targets=targets,
                token_indices=args[key]['token_indices'],
                grad_scale=grad_scale).images # ref_intermediate_latents=latents_list

    new_image.paste(outputs[1], (512*(idx_num+1), 0))


generation_image_path = os.path.join(output_dir_root, SOURCE_IMAGE_PATH.split("/")[-1])
if not os.path.exists(generation_image_path):
    os.makedirs(generation_image_path)
new_image.save(os.path.join(generation_image_path, target_prompt + '.png'))

source_images.append(Image.open(SOURCE_IMAGE_PATH).resize((256,256)))
generated_images.append(outputs[1].resize((256,256)))
eval_prompts.append(target_prompt)

print(os.path.join(generation_image_path, target_prompt + '.png'))
