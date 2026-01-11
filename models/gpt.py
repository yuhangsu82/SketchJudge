from __future__ import annotations
import os, re, base64, logging, time, random
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image
from openai import OpenAI
from .base import AbstractModel, register_model


def _img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _build_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = sample["query"]
    img_tokens = re.findall(r"<(image_\d+)>", query)
    text_parts = re.split(r"<image_\d+>", query)

    content: List[Dict[str, Any]] = []
    for idx, txt in enumerate(text_parts):
        if txt and txt.strip():
            content.append({"type": "text", "text": txt})
        if idx < len(img_tokens):
            key = img_tokens[idx]
            img_obj = sample.get(key)
            if img_obj is None:
                logging.warning("Missing image for %s", key)
                continue
            if isinstance(img_obj, (str, Path)):
                img_obj = Image.open(img_obj)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_img_to_b64(img_obj)}"}
            })

    return [{"role": "user", "content": content}]


@register_model("gpt")
class GPTModel(AbstractModel):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        retry_attempts: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _to_openai_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return _build_messages(sample)

    def _extract_text(self, resp) -> str:
        try:
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logging.exception("Failed to parse OpenAI response: %s", e)
            return ""

    def generate_from_sample(self, sample: Dict[str, Any]) -> str:
        messages = self._to_openai_messages(sample)

        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    # max_tokens=self.max_tokens,
                )
                return self._extract_text(resp)
            except Exception as e:
                if attempt >= self.retry_attempts:
                    logging.error("All retries failed: %s", e)
                    break
                sleep_s = min(30, 2 ** (attempt - 1) + random.random())
                logging.warning(f"Attempt {attempt} failed: {e}. Retry in {sleep_s:.1f}s.")
                time.sleep(sleep_s)

        return ""
