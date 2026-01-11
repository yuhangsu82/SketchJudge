from __future__ import annotations
import re, base64, logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List
import torch, transformers
from PIL import Image

# Optional dependency – required only for image/video prompts
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(msgs): 
        return None, None

from .base import AbstractModel, register_model


def _img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert sample with <image_n> placeholders to Qwen chat format."""
    query = sample["query"]
    img_tokens = re.findall(r"<(image_\d+)>", query)
    text_parts = re.split(r"<image_\d+>", query)

    content: List[Dict[str, str]] = []
    for idx, txt in enumerate(text_parts):
        if txt.strip():
            content.append({"type": "text", "text": txt})
        if idx < len(img_tokens):
            key = img_tokens[idx]
            img_obj = sample.get(key)
            if img_obj is None:
                logging.warning("Missing image for %s", key)
                continue
            if isinstance(img_obj, (str, Path)):
                img_obj = Image.open(img_obj)
            content.append({"type": "image",
                            "image": f"data:image/png;base64,{_img_to_b64(img_obj)}"})

    return [{"role": "user", "content": content}]


@register_model("qwen")
class QwenModel(AbstractModel):
    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.processor = transformers.AutoProcessor.from_pretrained(model_path, max_pixels=512 * 28 * 28)
        self.model = transformers.AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate_from_sample(self, sample: Dict[str, Any]) -> str:
        msgs = _build_messages(sample)
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        imgs, vids = process_vision_info(msgs)
        inputs = self.processor(
            text=[prompt],
            images=imgs,
            videos=vids,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        vis_counts = []
        if "image_grid_thw" in inputs:
            for thw in inputs["image_grid_thw"]:
                t, h, w = [int(x) for x in thw]
                vis_counts.append(t * h * w)
        
        print(f"Vision counts: {vis_counts}")

        outs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        gen = outs[:, inputs.input_ids.shape[-1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
