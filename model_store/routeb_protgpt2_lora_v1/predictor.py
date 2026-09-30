"""Loader/inference wrapper for the Route B conditional ProtGPT2 generator.

Generation only — no training. Loads the frozen ProtGPT2 base in 4-bit and
attaches this package's pre-trained LoRA adapter (see README.md)."""

from __future__ import annotations

import re
from pathlib import Path

from common.model_sync import sync_model_weights, weights_dir_for

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

BASE_MODEL_NAME = "nferruz/ProtGPT2"
EOT = "<|endoftext|>"
TAG_TOKENS = ["<AMP>", "<ANTIBACTERIAL>", "<ANTIBIOFILM>"]
TAG_ORDER = TAG_TOKENS  # fixed order the adapter was trained on

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")
VALID_RESIDUE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


class ProtGPT2Generator:
    """Lazy-loaded ProtGPT2 (4-bit NF4) + LoRA adapter. Samples peptides
    conditioned on a subset of TAG_TOKENS. See README.md for architecture."""

    def __init__(self, model_dir: Path = MODEL_DIR, base_model_name: str = BASE_MODEL_NAME):
        self.model_dir = Path(model_dir)
        self.base_model_name = base_model_name
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # float16, not bfloat16 -- bfloat16 has no native hardware support
            # on T4 (Turing, compute capability 7.5; bfloat16 needs Ampere+).
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        base_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.loaded = True

    @staticmethod
    def _extract_sequence(decoded: str, min_length: int, max_length: int) -> str | None:
        # decoded looks like "<|endoftext|><AMP> M K ... K<|endoftext|><|endoftext|>...".
        # The leading EOT is the prompt's own start-of-sequence marker, not a
        # generated stop -- split it off first, then take the chunk up to the
        # *next* EOT (the one the model actually generated, or all remaining
        # text if it ran out of max_new_tokens without emitting one).
        segments = decoded.split(EOT)
        body = segments[1] if segments and segments[0] == "" else segments[0]
        for tag in TAG_TOKENS:
            body = body.replace(tag, "")
        sequence = "".join(body.split())
        if not (min_length <= len(sequence) <= max_length):
            return None
        if not VALID_RESIDUE_RE.match(sequence):
            return None
        return sequence

    def generate(
        self,
        n_peptides: int,
        tags: list[str] | None = None,
        min_length: int = 6,
        max_length: int = 50,
        max_new_tokens: int = 120,
        batch_size: int = 16,
        max_attempts: int = 20,
        top_k: int = 950,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
    ) -> list[str]:
        """Sample up to n_peptides unique, validated peptide sequences
        conditioned on `tags` (a subset of TAG_TOKENS, e.g. ["<AMP>"])."""
        self._load()
        torch = self.torch
        tags = tags or ["<AMP>"]
        unknown = set(tags) - set(TAG_TOKENS)
        if unknown:
            raise ValueError(f"Unknown generation tags: {sorted(unknown)}; expected a subset of {TAG_TOKENS}")

        prompt = EOT + "".join(tag for tag in TAG_ORDER if tag in tags) + " "
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        sequences: list[str] = []
        seen: set[str] = set()

        with torch.no_grad():
            for _ in range(max_attempts):
                if len(sequences) >= n_peptides:
                    break
                batch_prompt_ids = prompt_ids.repeat(batch_size, 1)
                output_ids = self.model.generate(
                    batch_prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids(EOT),
                )
                decoded_batch = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
                for decoded in decoded_batch:
                    sequence = self._extract_sequence(decoded, min_length, max_length)
                    if sequence is None or sequence in seen:
                        continue
                    seen.add(sequence)
                    sequences.append(sequence)
                    if len(sequences) >= n_peptides:
                        break

        return sequences[:n_peptides]
