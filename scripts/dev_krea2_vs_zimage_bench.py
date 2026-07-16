"""Benchmark: Krea2_Turbo vs Rayzist_bf16 (Z-Image) across 3 prompts × 3 aspect ratios.

Not a pytest (needs GPU + full weights for both packs). Run manually on a CUDA box:
    .venv/Scripts/python.exe scripts/dev_krea2_vs_zimage_bench.py             # both packs
    .venv/Scripts/python.exe scripts/dev_krea2_vs_zimage_bench.py --pack krea2
    .venv/Scripts/python.exe scripts/dev_krea2_vs_zimage_bench.py --pack zimage
    .venv/Scripts/python.exe scripts/dev_krea2_vs_zimage_bench.py --report    # print table

Design
- 3 prompts (photorealistic portrait, anime, oil painting), each assigned a distinct aspect
  ratio (9:16 portrait / 1:1 square / 16:9 landscape) that fits the subject.
- 2 packs (Krea2_Turbo fp8_krea, Rayzist_bf16 diffusers).
- 2 runs per (pack, prompt): first = COLD (pipeline just loaded), second = WARM.
- Deterministic seeds shared across packs so outputs are comparable.
- Timings persisted incrementally to outputs/bench/results.json so a crash on one pack
  doesn't lose the other pack's numbers. When both packs have data, --report prints a
  markdown comparison table.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "outputs" / "bench"
RESULTS_JSON = BENCH_DIR / "results.json"


PROMPTS = [
    {
        "id": "portrait",
        "label": "photorealistic portrait",
        "prompt": (
            "a photorealistic portrait of a woman in her 30s with auburn hair, "
            "soft window light, shallow depth of field, high detail, 85mm lens"
        ),
        "width": 768,
        "height": 1344,
        "aspect": "9:16 portrait",
    },
    {
        "id": "anime",
        "label": "anime illustration",
        "prompt": (
            "anime illustration of a young mage casting a fireball, dynamic pose, "
            "vibrant colors, cel shaded, detailed background of ancient library"
        ),
        "width": 1024,
        "height": 1024,
        "aspect": "1:1 square",
    },
    {
        "id": "oil",
        "label": "oil painting",
        "prompt": (
            "an oil painting of a stormy seascape at dusk, dramatic clouds, "
            "crashing waves, visible brush strokes, Turner style"
        ),
        "width": 1344,
        "height": 768,
        "aspect": "16:9 landscape",
    },
]


PACKS = {
    "krea2": {
        "id": "krea2",
        "name": "Krea2_Turbo",
        "path": ROOT / "models" / "packs" / "Krea2_Turbo" / "modelpack.yaml",
        "steps": 8,
        "guidance": 0.0,
    },
    "zimage": {
        "id": "zimage",
        "name": "Rayzist_bf16",
        "path": ROOT / "models" / "packs" / "Rayzist_bf16" / "modelpack.yaml",
        "steps": 8,
        "guidance": 0.0,
    },
}


SEED_BASE = 1234


def _load_results() -> dict:
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {}


def _save_result(pack_id: str, entry: dict) -> None:
    data = _load_results()
    data.setdefault(pack_id, []).append(entry)
    RESULTS_JSON.write_text(json.dumps(data, indent=2))


def _reset_pack_results(pack_id: str) -> None:
    data = _load_results()
    if pack_id in data:
        del data[pack_id]
    RESULTS_JSON.write_text(json.dumps(data, indent=2))


def _free_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class _Settings:
    def __init__(self, profile):
        self.runtime_profile = profile
        self.resource_tier = profile

        class _Ctl:
            def current(_self):
                return profile

        self.resource_tier_controller = _Ctl()


def _bench_pack_inproc(pack_id: str, tier: str = "constrained", results_key: str | None = None) -> None:
    """Run one pack's 6 gens in the current process, persisting each result."""
    import torch

    sys.path.insert(0, str(ROOT))
    from app.config.profiles import RUNTIME_PROFILES
    from app.core.backends import create_backend
    from app.core.model_registry import load_model_pack
    from app.core.worker.types import GenerationRequest

    pack_cfg = PACKS[pack_id]
    save_key = results_key or pack_id
    print(f"\n=== {pack_cfg['name']} (tier={tier}, key={save_key}) ===", flush=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    _reset_pack_results(save_key)

    pack = load_model_pack(pack_cfg["path"])
    profile = RUNTIME_PROFILES[tier]
    settings = _Settings(profile)
    backend = create_backend(settings=settings, model_pack=pack, resource_tier=profile)
    print(f"backend: {type(backend).__name__}", flush=True)

    t_load = time.time()
    backend._ensure_loaded()
    load_s = time.time() - t_load
    print(f"pipeline loaded in {load_s:.1f}s", flush=True)
    _free_cuda()
    _save_result(save_key, {"event": "load", "duration_s": load_s, "tier": tier})

    for i, spec in enumerate(PROMPTS):
        seed = SEED_BASE + i
        for _run_idx, temp in enumerate(("cold", "warm")):
            req = GenerationRequest(
                prompt=spec["prompt"],
                width=spec["width"],
                height=spec["height"],
                steps=pack_cfg["steps"],
                guidance_scale=pack_cfg["guidance"],
                seed=seed,
            )
            t0 = time.time()
            result = backend.generate(req)
            dt = time.time() - t0
            out_path = BENCH_DIR / f"{save_key}_{spec['id']}_{temp}.png"
            result.image.save(out_path)
            print(
                f"  [{spec['id']:8s}] {temp:4s} {spec['width']}x{spec['height']} "
                f"seed={seed} steps={result.steps} in {dt:.1f}s -> {out_path.name}",
                flush=True,
            )
            _save_result(
                save_key,
                {
                    "event": "gen",
                    "prompt": spec["id"],
                    "temp": temp,
                    "duration_s": dt,
                    "width": spec["width"],
                    "height": spec["height"],
                    "seed": seed,
                    "backend": result.backend,
                    "steps": result.steps,
                    "tier": tier,
                },
            )


def _spawn_pack_subprocess(pack_id: str) -> int:
    """Run one pack in a fresh subprocess so segfaults don't kill sibling packs."""
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--pack",
        pack_id,
        "--inproc",
    ]
    print(f"\n>>> spawning subprocess for pack '{pack_id}'", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _print_table() -> None:
    data = _load_results()
    if not data:
        print("no results yet")
        return
    print("\n## Benchmark results (durations in seconds)\n", flush=True)

    def lookup(pack_id: str, prompt_id: str, temp: str) -> float | None:
        for entry in data.get(pack_id, []):
            if entry.get("event") == "gen" and entry["prompt"] == prompt_id and entry["temp"] == temp:
                return entry["duration_s"]
        return None

    have_krea = "krea2" in data
    have_zimage = "zimage" in data

    # Table 1: per-pack cold vs warm
    for pack_id in ("krea2", "krea2_high", "krea2_high_opt", "zimage", "zimage_opt"):
        if pack_id not in data:
            continue
        base_pack = "krea2" if pack_id.startswith("krea2") else "zimage"
        pack_name = PACKS[base_pack]["name"]
        load_entry = next((e for e in data[pack_id] if e.get("event") == "load"), None)
        tier = load_entry.get("tier", "constrained") if load_entry else "constrained"
        load_note = f"  (tier={tier}, pipeline load: {load_entry['duration_s']:.1f}s)" if load_entry else ""
        print(f"### {pack_name} [{pack_id}]{load_note}\n")
        print("| prompt   | aspect          | seed | cold (s) | warm (s) |")
        print("|----------|-----------------|------|----------|----------|")
        for spec in PROMPTS:
            cold = lookup(pack_id, spec["id"], "cold")
            warm = lookup(pack_id, spec["id"], "warm")
            seed = SEED_BASE + PROMPTS.index(spec)
            cold_s = f"{cold:.1f}" if cold is not None else "—"
            warm_s = f"{warm:.1f}" if warm is not None else "—"
            print(f"| {spec['id']:8s} | {spec['aspect']:15s} | {seed} | {cold_s:>8s} | {warm_s:>8s} |")
        print()

    # Table 2: side-by-side comparison
    if have_krea and have_zimage:
        print("### Head-to-head (warm timings)\n")
        print("| prompt   | aspect          | krea2 warm (s) | zimage warm (s) | krea2 / zimage |")
        print("|----------|-----------------|----------------|-----------------|----------------|")
        for spec in PROMPTS:
            k = lookup("krea2", spec["id"], "warm")
            z = lookup("zimage", spec["id"], "warm")
            ratio = f"{k/z:.2f}×" if (k and z) else "—"
            k_s = f"{k:.1f}" if k is not None else "—"
            z_s = f"{z:.1f}" if z is not None else "—"
            print(f"| {spec['id']:8s} | {spec['aspect']:15s} | {k_s:>14s} | {z_s:>15s} | {ratio:>14s} |")
        print()

    # Table: Krea2 high vs high+opt (compile + sage)
    if "krea2_high" in data and "krea2_high_opt" in data:
        print("### Krea2 high tier: baseline vs optimizations (warm timings)\n")
        print("| prompt   | aspect          | high (s) | high+opt (s) | speedup |")
        print("|----------|-----------------|----------|--------------|---------|")
        for spec in PROMPTS:
            base = lookup("krea2_high", spec["id"], "warm")
            opt = lookup("krea2_high_opt", spec["id"], "warm")
            ratio = f"{base/opt:.2f}×" if (base and opt) else "—"
            b_s = f"{base:.1f}" if base is not None else "—"
            o_s = f"{opt:.1f}" if opt is not None else "—"
            print(f"| {spec['id']:8s} | {spec['aspect']:15s} | {b_s:>8s} | {o_s:>12s} | {ratio:>7s} |")
        print()

    # Table: Z-Image baseline vs optimizations
    if "zimage" in data and "zimage_opt" in data:
        print("### Z-Image: baseline vs optimizations (warm timings)\n")
        print("| prompt   | aspect          | baseline (s) | opt (s) | speedup |")
        print("|----------|-----------------|--------------|---------|---------|")
        for spec in PROMPTS:
            base = lookup("zimage", spec["id"], "warm")
            opt = lookup("zimage_opt", spec["id"], "warm")
            ratio = f"{base/opt:.2f}×" if (base and opt) else "—"
            b_s = f"{base:.1f}" if base is not None else "—"
            o_s = f"{opt:.1f}" if opt is not None else "—"
            print(f"| {spec['id']:8s} | {spec['aspect']:15s} | {b_s:>12s} | {o_s:>7s} | {ratio:>7s} |")
        print()

    # Table 3: Krea2 constrained vs high tier
    if "krea2" in data and "krea2_high" in data:
        print("### Krea2 constrained vs high tier (warm timings)\n")
        print("| prompt   | aspect          | constrained (s) | high (s) | speedup |")
        print("|----------|-----------------|-----------------|----------|---------|")
        for spec in PROMPTS:
            c = lookup("krea2", spec["id"], "warm")
            h = lookup("krea2_high", spec["id"], "warm")
            ratio = f"{c/h:.2f}×" if (c and h) else "—"
            c_s = f"{c:.1f}" if c is not None else "—"
            h_s = f"{h:.1f}" if h is not None else "—"
            print(f"| {spec['id']:8s} | {spec['aspect']:15s} | {c_s:>15s} | {h_s:>8s} | {ratio:>7s} |")
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=["krea2", "zimage"], default=None)
    parser.add_argument("--inproc", action="store_true", help="Run in this process (default: spawn subprocess).")
    parser.add_argument("--report", action="store_true", help="Print the results table and exit.")
    parser.add_argument("--tier", default="constrained", choices=["high", "balanced", "constrained"])
    parser.add_argument("--key", default=None, help="Override the results.json key (e.g. 'krea2_high').")
    args = parser.parse_args()

    if args.report:
        _print_table()
        return 0

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    if args.pack and args.inproc:
        _bench_pack_inproc(args.pack, tier=args.tier, results_key=args.key)
        return 0

    packs_to_run = [args.pack] if args.pack else ["krea2", "zimage"]
    for pack_id in packs_to_run:
        rc = _spawn_pack_subprocess(pack_id)
        if rc != 0:
            print(f"!! pack '{pack_id}' subprocess exited with code {rc}", flush=True)

    _print_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
