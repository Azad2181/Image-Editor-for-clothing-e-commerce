# Image-Editor-for-clothing-e-commerce.
Image Editor for a clothing brand. This is a confidential project, so I cannot disclose the complete source code. However, the editing workflow, architecture, and implementation approach can be adapted and customized to meet your company's specific requirements.

==========================================

## 1. Requirements

- Python 3.10+
- **GPU strongly recommended** (NVIDIA with 6GB+ VRAM). It will run on CPU
  too, but each edit can take several minutes instead of a few seconds.
- ~6 GB free disk space (model weights download on first run)

## 2. Setup (in VS Code)

Open this folder in VS Code, then in the integrated terminal:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> If you have an NVIDIA GPU, make sure you install the CUDA-enabled build of
> PyTorch for best speed. See https://pytorch.org/get-started/locally/ and
> pick the command matching your CUDA version, then `pip install -r requirements.txt`
> for the rest.

## 3. Run the interactive app (single image, try-it-out UI)

```bash
python app.py
```

This starts a local Gradio web app (usually at `http://127.0.0.1:7860`).
Open it in your browser, upload a product photo, type an instruction, and
click **Edit Image**. The first run will download the model (~5GB) — after
that it's cached locally and loads fast.

### Example instructions to try
- `change the background to a clean white studio backdrop`
- `put the model on a sunny beach`
- `make the background a minimalist grey gradient studio`
- `make the lighting brighter and more professional`

## 4. Run batch mode (edit a whole folder of product photos at once)

Useful once you've found an instruction that works well and want to apply it
to your full catalog:

```bash
python batch_edit.py \
  --input_dir ./raw_photos \
  --output_dir ./edited_photos \
  --instruction "change background to white studio" \
  --steps 25
```

Edited images are saved as `<original_name>_edited.png` in `--output_dir`.

## 5. Tuning results

- **image_guidance_scale** (1.0–2.5): higher = keeps the edited image closer
  to the original (garment shape/color stays truer). Lower = allows more
  dramatic changes.
- **text_guidance_scale** (1.0–15.0): higher = follows your text instruction
  more strongly.
- **steps** (10–50): more steps = higher quality but slower.

If edits are too weak, raise `text_guidance_scale` or lower
`image_guidance_scale`. If the garment itself is changing shape/color when
it shouldn't, raise `image_guidance_scale`.

## 6. Notes on this model's strengths/limits

- Best for: background changes, lighting/mood changes, style tweaks, simple
  scene changes.
- Weaker at: precise garment shape edits, adding text/logos, or completely
  replacing the person/model in the photo. For those, look into
  `Qwen/Qwen-Image-Edit` or `black-forest-labs/FLUX.1-Kontext` (heavier
  models, need more VRAM) as a next step.
- This is free and runs fully on your own machine — no data leaves your
  computer, no API costs.

## 7. License note

`timbrooks/instruct-pix2pix` weights are released for research use; check
the model card on Hugging Face for the latest license terms before using
output commercially on your live website, and consider testing
`black-forest-labs/FLUX.1-schnell`-based editing tools (Apache 2.0) if you
need a fully commercial-safe license down the line.
