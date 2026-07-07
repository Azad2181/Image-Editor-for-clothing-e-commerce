"""
Product Image Editor — Clothing Brand
--------------------------------------
Upload a raw product photo, describe the edit in plain text, and get an
edited image back. Powered by the free, open-source InstructPix2Pix model
(timbrooks/instruct-pix2pix) via Hugging Face Diffusers.

Run:
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import gradio as gr

MODEL_ID = "timbrooks/instruct-pix2pix"


def get_device_and_dtype():
    """Pick the best available device automatically."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps", torch.float32
    return "cpu", torch.float32


DEVICE, DTYPE = get_device_and_dtype()
print(f"[INFO] Using device: {DEVICE} (dtype: {DTYPE})")

print("[INFO] Loading InstructPix2Pix pipeline (first run will download ~5GB)...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,  # remove if you want the built-in NSFW filter
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

# Memory-saving tricks — helpful on smaller GPUs / CPU
if DEVICE == "cuda":
    pipe.enable_attention_slicing()
print("[INFO] Model loaded. Ready to edit images.")


def edit_image(
    input_image: Image.Image,
    instruction: str,
    image_guidance_scale: float,
    text_guidance_scale: float,
    steps: int,
    seed: int,
):
    if input_image is None:
        raise gr.Error("Please upload a product image first.")
    if not instruction or not instruction.strip():
        raise gr.Error("Please type an edit instruction, e.g. 'change background to white studio'.")

    input_image = input_image.convert("RGB")

    # Resize so longest side is 768px — good quality/speed tradeoff
    max_side = 768
    w, h = input_image.size
    scale = max_side / max(w, h)
    if scale < 1:
        input_image = input_image.resize((int(w * scale), int(h * scale)))

    generator = torch.manual_seed(int(seed)) if seed is not None else None

    result = pipe(
        instruction,
        image=input_image,
        num_inference_steps=int(steps),
        image_guidance_scale=image_guidance_scale,
        guidance_scale=text_guidance_scale,
        generator=generator,
    )
    return result.images[0]


EXAMPLE_INSTRUCTIONS = [
    "change the background to a clean white studio backdrop",
    "put the model on a sunny beach",
    "make the background a minimalist grey gradient studio",
    "change the background to an urban street at golden hour",
    "make the lighting brighter and more professional",
]

with gr.Blocks(title="Product Image Editor") as demo:
    gr.Markdown(
        """
        # 👕 Clothing Product Image Editor
        Upload a raw product photo and describe the edit you want in plain text.
        Powered by the free **InstructPix2Pix** model (runs locally, no API key needed).
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Raw Product Image", height=400)
            instruction = gr.Textbox(
                label="Edit Instruction",
                placeholder="e.g. change background to white studio",
            )
            gr.Examples(examples=EXAMPLE_INSTRUCTIONS, inputs=instruction)

            with gr.Accordion("Advanced settings", open=False):
                image_guidance_scale = gr.Slider(
                    1.0, 2.5, value=1.5, step=0.05,
                    label="Image Guidance Scale (higher = stays closer to original image)",
                )
                text_guidance_scale = gr.Slider(
                    1.0, 15.0, value=7.5, step=0.5,
                    label="Text Guidance Scale (higher = follows instruction more strongly)",
                )
                steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
                seed = gr.Number(value=42, label="Seed (for reproducibility)")

            run_btn = gr.Button("✨ Edit Image", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Edited Result", height=400)

    run_btn.click(
        fn=edit_image,
        inputs=[input_image, instruction, image_guidance_scale, text_guidance_scale, steps, seed],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
