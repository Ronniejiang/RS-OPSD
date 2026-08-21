"""OpenAI-compatible model adapter matching OPD-V's serving interface."""

from __future__ import annotations

import base64
from io import BytesIO
import threading
import time

from PIL import Image


class OPDVOpenAIGenerator:
    """Generate one answer at a time through an OpenAI-compatible vision endpoint.

    The request shape, retry policy, and optional ``enable_thinking`` argument
    intentionally follow ``other_methods/OPD-V/eval/infer.py``.  A client is
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
        """Strip Qwen thinking/answer wrappers used by OPD-V serving."""
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

    def image_to_data_uri(self, image: Image.Image) -> str:
        """Convert a PIL image to an API-safe URI, bounding its pixel count."""
        image = image.convert("RGB")
        width, height = image.size
        pixel_count = width * height
        if pixel_count > self.max_pixels:
            scale = (self.max_pixels / pixel_count) ** 0.5
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )

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
