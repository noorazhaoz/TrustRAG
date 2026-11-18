"""TrustRAG MLX generation backend (Apple Silicon).

Keep the previous Transformers / Qwen2.5-0.5B implementation unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


@dataclass
class GenerationResult:
    answer: str
    model_id: str
    max_tokens: int
    generation_tokens: int
    finish_reason: Optional[str]


class MLXBackend:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        # Import lazily so importing this module does not load a 4B model.
        from mlx_lm import load

        self.model_id = model_id
        print(f"Loading MLX model: {model_id}")
        self.model, self.tokenizer = load(model_id)

    def generate(
        self,
        question: str,
        evidence: str,
        *,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        if system_prompt is None:
            system_prompt = (
                "You answer questions using only the provided evidence. "
                "Be direct and concise. If the evidence does not contain "
                "the information needed, say that the evidence is insufficient. "
                "Do not invent numbers, sources, or facts."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Evidence:\n{evidence}\n\nQuestion: {question}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


        from mlx_lm import stream_generate

        pieces = []
        final_response = None

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            pieces.append(response.text)
            final_response = response

        answer = "".join(pieces).strip()

        generation_tokens = (
            final_response.generation_tokens
            if final_response is not None
            else 0
        )

        finish_reason = (
            final_response.finish_reason
            if final_response is not None
            else None
        )

        print(
            f"[Qwen] max_tokens={max_tokens}, "
            f"generated={generation_tokens}, "
            f"finish_reason={finish_reason}",
            flush=True,
        )

        return GenerationResult(
            answer=answer,
            model_id=self.model_id,
            max_tokens=max_tokens,
            generation_tokens=generation_tokens,
            finish_reason=finish_reason,
        )

