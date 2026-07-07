"""
Batch Product Image Editor
---------------------------
Apply the SAME edit instruction to every image in a folder — useful when
you want to give all product photos a consistent background/style.

Usage:
    python batch_edit.py --input_dir ./raw_photos --output_dir ./edited_photos \
        --instruction "change background to white studio" --steps 25
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

MODEL_ID = "timbrooks/instruct-pix2pix"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser(description="Batch edit product images with InstructPix2Pix")
    parser.add_argument("--input_dir", required=True, help="Folder of raw product photos")
    parser.add_argument("--output_dir", required=True, help="Folder to save edited photos")
    parser.add_argument("--instruction", required=True, help="Edit instruction, e.g. 'white studio background'")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--text_guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_side", type=int, default=768, help="Resize longest side to this many pixels")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading InstructPix2Pix pipeline...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()

    image_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    )
    if not image_files:
        print(f"[WARN] No images found in {input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images. Starting batch edit...")

    for i, img_path in enumerate(image_files, start=1):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        scale = args.max_side / max(w, h)
        if scale < 1:
            image = image.resize((int(w * scale), int(h * scale)))

        generator = torch.manual_seed(args.seed)
        result = pipe(
            args.instruction,
            image=image,
            num_inference_steps=args.steps,
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.text_guidance_scale,
            generator=generator,
        ).images[0]

        out_path = output_dir / f"{img_path.stem}_edited.png"
        result.save(out_path)
        print(f"[{i}/{len(image_files)}] Saved -> {out_path}")

    print("[DONE] Batch editing complete.")


if __name__ == "__main__":
    main()
