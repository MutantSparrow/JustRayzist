from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test whether a pack's text_encoder can run as a local causal LLM "
            "for prompt rewriting."
        )
    )
    parser.add_argument(
        "--pack",
        default="Rayzist_bf16",
        help="Pack folder name under models/packs (default: Rayzist_bf16).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override model directory (defaults to <pack>/config/text_encoder).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Override tokenizer directory (defaults to <pack>/config/tokenizer).",
    )
    parser.add_argument(
        "--prompt",
        default="portrait of a woman",
        help="Original prompt to rewrite/expand.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a prompt rewriting assistant for text-to-image generation. "
            "Rewrite the prompt to be vivid and specific while preserving intent. "
            "Return only one rewritten prompt."
        ),
        help="System instruction used for rewriting.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Max generated tokens (default: 96).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype hint (default: auto).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from local files if required.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling; use greedy decoding.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only test load/tokenization path, skip generation.",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = _resolve_root()
    pack_dir = root / "models" / "packs" / args.pack
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack directory not found: {pack_dir}")

    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else pack_dir / "config" / "text_encoder"
    tokenizer_dir = (
        Path(args.tokenizer_dir).expanduser().resolve() if args.tokenizer_dir else pack_dir / "config" / "tokenizer"
    )
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    return pack_dir, model_dir, tokenizer_dir


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _resolve_dtype(requested: str, device: str, torch_module: Any) -> Any:
    if requested == "float32":
        return torch_module.float32
    if requested == "float16":
        return torch_module.float16
    if requested == "bfloat16":
        return torch_module.bfloat16
    if device == "cuda":
        return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16
    return torch_module.float32


def _make_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    user_message = (
        "Rewrite this image prompt. Keep intent, increase visual specificity, "
        "and return only the rewritten prompt.\n\n"
        f"Original prompt: {user_prompt}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except TypeError:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                pass
        except Exception:
            pass
    return f"{system_prompt}\n\n{user_message}\n\nRewritten prompt:"


def _load_causal_model(
    *,
    model_dir: Path,
    dtype: Any,
    trust_remote_code: bool,
    auto_model_for_causal_lm: Any,
) -> Any:
    common_kwargs = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    try:
        return auto_model_for_causal_lm.from_pretrained(
            str(model_dir),
            dtype=dtype,
            **common_kwargs,
        )
    except TypeError:
        return auto_model_for_causal_lm.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            **common_kwargs,
        )


def _extract_rewritten_text(full_text: str, input_text: str) -> str:
    candidate = full_text[len(input_text) :].strip() if full_text.startswith(input_text) else full_text.strip()
    # Trim optional chain-of-thought block if the model emits it.
    candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL).strip()
    if "Rewritten prompt:" in candidate:
        candidate = candidate.split("Rewritten prompt:", 1)[-1].strip()
    return candidate


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    try:
        pack_dir, model_dir, tokenizer_dir = _resolve_paths(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except Exception as exc:
        print(f"[ERROR] Missing runtime dependency: {exc}")
        return 2

    transformers_logging.set_verbosity_error()

    device = _resolve_device(args.device, torch)
    dtype = _resolve_dtype(args.dtype, device, torch)

    print("=== Text Encoder LLM Smoke Test ===")
    print(f"Pack: {pack_dir.name}")
    print(f"Model dir: {model_dir}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Offline flags: HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')} TRANSFORMERS_OFFLINE={os.getenv('TRANSFORMERS_OFFLINE')}")

    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
        model = _load_causal_model(
            model_dir=model_dir,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            auto_model_for_causal_lm=AutoModelForCausalLM,
        )
        if device == "cuda":
            model = model.to("cuda")
        model.eval()
    except Exception as exc:
        print("[FAIL] Could not load text_encoder with AutoModelForCausalLM.")
        print(f"[DETAIL] {exc}")
        print(
            "Interpretation: prompt rewriting via text_encoder causal LM is likely not directly usable "
            "with this local config/weights combination."
        )
        return 1

    load_ms = int((time.time() - t0) * 1000)
    print(f"[OK] Loaded tokenizer={tokenizer.__class__.__name__} model={model.__class__.__name__} in {load_ms} ms")

    prompt_text = _make_prompt(tokenizer, args.system, args.prompt)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = int(encoded["input_ids"].shape[-1])
    print(f"Input tokens: {input_len}")

    if args.load_only:
        print("[OK] Load-only mode complete.")
        return 0

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    do_sample = not args.no_sample
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)

    t1 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **gen_kwargs)
    gen_ms = int((time.time() - t1) * 1000)

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    input_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
    rewritten = _extract_rewritten_text(full_text, input_text)

    print(f"Generation time: {gen_ms} ms")
    print("--- rewritten prompt ---")
    print(rewritten or "(empty)")
    print("------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
