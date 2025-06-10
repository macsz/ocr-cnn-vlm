import logging

import torch
from rich.logging import RichHandler
from transformers import pipeline

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("vlm")


def init_pipeline(model_name, device="cuda", torch_dtype=torch.bfloat16):
    logger.info(f"Loading model {model_name}...")
    pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )

    return pipe


def create_message(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Act as an OCR assistant. Analyze the provided image and:\n"
                    "- Identify and transcribe all visible text in the image exactly as it appears.\n"
                    "- Preserve the original line breaks, spacing, and formatting from the image.\n"
                    "- Output only the transcribed text, line by line, without adding any commentary or explanations or special characters.\n"
                    "- Do not include any additional information or context.\n",
                },
                {"type": "image", "url": str(image_path)},
            ],
        }
    ]

    return messages
