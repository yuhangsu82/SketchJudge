from __future__ import annotations
import re, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import transformers
from .base import AbstractModel, register_model


def _build_qianfan_prompt_and_images(sample: Dict[str, Any]) -> Tuple[str, List[Image.Image]]:
    """
    Convert a sample with <image_n> placeholders into:
      - a prompt string with <image> tokens
      - a list of PIL.Images aligned with <image> order
    """
    query = sample["query"]
    img_tokens = re.findall(r"<(image_\d+)>", query)
    text_parts = re.split(r"<image_\d+>", query)

    images: List[Image.Image] = []
    prompt_parts: List[str] = []

    for idx, txt in enumerate(text_parts):
        if txt and txt.strip():
            prompt_parts.append(txt)
        if idx < len(img_tokens):
            key = img_tokens[idx]
            img = sample.get(key)
            if img is None:
                logging.warning("Missing image for %s", key)
                continue
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            if not isinstance(img, Image.Image):
                logging.warning("Invalid image type for %s: %s", key, type(img))
                continue
            images.append(img.convert("RGB"))
            prompt_parts.append("<image>")

    return "".join(prompt_parts), images


def _vision_transform(size: int = 448):
    """
    Vision preprocessing fallback for Qianfan-VL-8B.

    This is used when AutoProcessor(images=...) is unavailable due to
    transformers version constraints. The configuration follows
    common CLIP/EVA/ViT conventions.
    """
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


@register_model("qianfan")
class QianfanModel(AbstractModel):
    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 1024,
        image_size: int = 448,
        dtype=torch.bfloat16,
    ):
        """
        Qianfan-VL-8B inference wrapper for benchmark evaluation.

        - Deterministic decoding (greedy)
        - Explicit vision preprocessing fallback
        - chat() API as the canonical inference path
        """
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.transform = _vision_transform(image_size)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = transformers.AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            dtype="auto",
        ).eval()

        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate_from_sample(self, sample: Dict[str, Any]) -> str:
        prompt, images = _build_qianfan_prompt_and_images(sample)
        if not images:
            logging.error("No images found for query: %s", sample.get("query", ""))
            return ""

        # Stack images into [N, C, H, W], aligned with <image> tokens
        pixel_values = torch.stack(
            [self.transform(im) for im in images], dim=0
        ).to(device=self.device, dtype=self.dtype)

        # Canonical Qianfan inference via chat()
        out_text = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config={
                "do_sample": False, # greedy decoding for evaluation
                "max_new_tokens": self.max_new_tokens,
            }
        )

        return out_text.strip() if isinstance(out_text, str) else str(out_text).strip()
