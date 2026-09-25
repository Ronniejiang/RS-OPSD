"""Vision-model adapters for OpenAI-compatible endpoints and local Transformers."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import threading
import time
from typing import Protocol

from PIL import Image


class AnswerGenerator(Protocol):
    """Minimal interface shared by all evaluator inference backends."""

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Generate one answer for an image and prompt."""


class OpenAICompatibleGenerator:
    """Generate one answer at a time through an OpenAI-compatible vision endpoint.

    The request shape, retry policy, and optional ``enable_thinking`` argument
    are compatible with vLLM and other OpenAI-compatible servers. A client is
    kept per worker thread because benchmark inference is concurrent.
    """

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model_id: str,
        max_tokens: int,
        max_retries: int,
        request_timeout: float,
        enable_thinking: bool | None,
        max_pixels: int,
        image_format: str,
        jpeg_quality: int,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.max_pixels = max_pixels
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self._thread_local = threading.local()

    @staticmethod
    def normalize_model_answer(model_answer_raw: str) -> str:
        """Strip common Qwen thinking and answer wrappers."""
        answer = model_answer_raw.strip()
        think_end = answer.rfind("</think>")
        if think_end != -1:
            answer = answer[think_end + len("</think>"):].strip()
        start = answer.rfind("<answer>")
        end = answer.find("</answer>", start + len("<answer>")) if start != -1 else -1
        if start != -1 and end > start:
            return answer[start + len("<answer>"):end].strip()
        if "Answer:" in answer:
            return answer[answer.find("Answer:"):].strip()
        return answer

    def _client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "OpenAI-compatible inference requires the 'openai' package. "
                    "Install it with: pip install openai"
                ) from error
            client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.request_timeout)
            self._thread_local.client = client
        return client

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Convert to RGB and bound image area consistently across backends."""
        image = image.convert("RGB")
        width, height = image.size
        pixel_count = width * height
        if pixel_count > self.max_pixels:
            scale = (self.max_pixels / pixel_count) ** 0.5
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    def image_to_data_uri(self, image: Image.Image) -> str:
        """Convert a PIL image to an API-safe URI, bounding its pixel count."""
        image = self.prepare_image(image)

        encoded = BytesIO()
        if self.image_format == "jpeg":
            image.save(encoded, format="JPEG", quality=self.jpeg_quality, optimize=True)
            mime_type = "image/jpeg"
        else:
            image.save(encoded, format="PNG", optimize=True)
            mime_type = "image/png"
        data = base64.b64encode(encoded.getvalue()).decode("ascii")
        return f"data:{mime_type};base64,{data}"

    def generate(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.image_to_data_uri(image)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        client = self._client()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                extra_kwargs = {}
                if self.enable_thinking is not None:
                    extra_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                    }
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    **extra_kwargs,
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The model response did not contain text content.")
                return self.normalize_model_answer(content)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(1.0)
        raise RuntimeError(f"API request failed after {self.max_retries} attempts: {last_error}")


class TransformersGenerator:
    """Local Hugging Face vision inference for hosts without a compatible vLLM.

    The evaluator keeps this backend deliberately single-worker: a model object
    is not a concurrent request server, and serial generation avoids duplicating
    the model or accumulating multiple high-resolution images on one GPU.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        max_tokens: int,
        enable_thinking: bool | None,
        max_pixels: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise ImportError(
                "Local Transformers inference requires both 'torch' and 'transformers'."
            ) from error

        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise ValueError(f"Not a Hugging Face model directory: {model_path}")
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.max_pixels = max_pixels
        self._torch = torch
        self._lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

    def prepare_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        pixel_count = width * height
        if pixel_count > self.max_pixels:
            scale = (self.max_pixels / pixel_count) ** 0.5
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    def generate(self, image: Image.Image, prompt: str) -> str:
        image = self.prepare_image(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        text = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")

        # One locked generation is required even if the caller accidentally
        # shares this object across threads.
        with self._lock, self._torch.inference_mode():
            inputs = inputs.to(self.model.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
            )
        continuation_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        output = self.processor.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return OpenAICompatibleGenerator.normalize_model_answer(output)


# Compatibility alias for earlier local launchers. New integrations should use
# OpenAICompatibleGenerator.
OPDVOpenAIGenerator = OpenAICompatibleGenerator
