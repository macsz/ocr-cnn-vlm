import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from enum import Enum
from glob import glob

import torch
from PIL import Image
from rich.console import Console
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer)

from common import create_message, init_pipeline, logger

model = None
tokenizer = None
pipe = None
processor = None


def resize_with_min_dimension(img, min_dimension):
    """
    Resize image so that no dimension is smaller than min_dimension while maintaining aspect ratio.

    Args:
        img: PIL Image to resize
        min_dimension: Minimum dimension (width or height) in pixels

    Returns:
        PIL Image resized to maintain aspect ratio with minimum dimension
    """
    width, height = img.size

    # Calculate scaling factor to ensure minimum dimension
    scale = max(min_dimension / width, min_dimension / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_img


class Category(Enum):
    ORIGINAL = "original"
    RAIN_LIGHT = "rain_light"
    RAIN_MEDIUM = "rain_medium"
    RAIN_HEAVY = "rain_heavy"
    FOG_LIGHT = "fog_light"
    FOG_MEDIUM = "fog_medium"
    FOG_HEAVY = "fog_heavy"
    RAIN_FOG_LIGHT = "rain_fog_light"
    RAIN_FOG_MEDIUM = "rain_fog_medium"
    RAIN_FOG_HEAVY = "rain_fog_heavy"


CATEGORIES = [
    Category.ORIGINAL,
    Category.RAIN_LIGHT,
    Category.RAIN_MEDIUM,
    Category.RAIN_HEAVY,
    Category.FOG_LIGHT,
    Category.FOG_MEDIUM,
    Category.FOG_HEAVY,
    Category.RAIN_FOG_LIGHT,
    Category.RAIN_FOG_MEDIUM,
    Category.RAIN_FOG_HEAVY,
]


class WeatherAugmentedImageDataset(Dataset):
    def __init__(self, img_dir, category: Category = Category.ORIGINAL):
        self.img_paths = []

        # Define transforms - normalize images for model input
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL image to tensor
            ]
        )

        self.img_paths.extend(glob(os.path.join(img_dir, f"*{category.value}*.jpg")))
        self.img_paths.extend(glob(os.path.join(img_dir, f"*{category.value}*.png")))

        logger.info(
            f"Found {len(self.img_paths)} images with category {category.value}."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        return {"path": img_path}


class WeatherAugmentedCroppedImageDataset(Dataset):
    def __init__(self, img_dir, category: Category = Category.ORIGINAL):
        self.img_paths = []

        # Define transforms - normalize images for model input
        self.transform = transforms.Compose(
            [
                # transforms.Lambda(lambda img: resize_with_min_dimension(img, 28)),
                transforms.ToTensor(),  # Convert PIL image to tensor
            ]
        )
        extensions = ["jpg", "png"]
        for ext in extensions:
            self.img_paths.extend(
                glob(os.path.join(img_dir, f"*{category.value}*.{ext}"))
            )

        if "fog" in category.value and "rain" not in category.value:
            self.img_paths = [path for path in self.img_paths if "rain" not in path]

        logger.info(
            f"Found {len(self.img_paths)} images with category {category.value}."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        return {"path": img_path}


def get_original_filename_svt(augmented_path):
    """Extract the original filename without augmentation info"""
    # Extract base filename without path and augmentation pattern
    filename = os.path.basename(augmented_path)
    # Extract just the initial part (00_00) from filename like 00_00_rain_light.jpg
    match = re.match(r"(\d+_\d+).*?\.(jpg|png)", filename)
    if match:
        return f"img/{match.group(1)}.jpg"
    return None


def get_original_filename_icdar(augmented_path):
    """Extract the original filename. Augmented path is like ch4_test_images/img_1_original.jpg,
    while the original path is like ch4_test_images/img_1.jpg"""
    filename = os.path.basename(augmented_path)
    # Extract just the initial part (img_1) from filename like img_1_original.jpg
    match = re.match(r"img_(\d+).*?\.(jpg|png)", filename)
    if match:
        return f"ch4_test_images/img_{match.group(1)}.jpg"
    logger.warning(f"Could not extract original filename from {augmented_path}")
    return None


def load_annotations_svt(xml_path):
    """Load annotations from XML file into a dictionary with original filenames as keys"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}

    for image in root.findall(".//image"):
        image_name = image.find("imageName").text
        tags = []
        for rect in image.findall(".//taggedRectangle"):
            tag = rect.find("tag").text
            x = int(rect.get("x"))
            y = int(rect.get("y"))
            width = int(rect.get("width"))
            height = int(rect.get("height"))
            tags.append({"text": tag, "bbox": [x, y, width, height]})
        annotations[image_name] = tags

    return annotations


def load_annotations_icdar(txt_path):
    """Load annotations from ICDAR 2015 format txt file."""
    annotations = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            image_path = parts[0]

            # The JSON part can be tricky if it's not valid.
            try:
                ann_json = json.loads(parts[1])
            except json.JSONDecodeError:
                logger.warning(f"Warning: Could not decode JSON for {image_path}")
                continue

            # image_name = os.path.basename(image_path)

            tags = []
            for ann in ann_json:
                transcription = ann["transcription"]
                if transcription == "###":
                    continue
                transcription = transcription.strip("#.,!?:;()-\"'").lower()

                points = ann["points"]
                # points is a list of lists, e.g., [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

                all_x = [p[0] for p in points]
                all_y = [p[1] for p in points]

                x = min(all_x)
                y = min(all_y)
                width = max(all_x) - x
                height = max(all_y) - y

                tags.append({"text": transcription, "bbox": [x, y, width, height]})

            annotations[image_path] = tags

    return annotations


def eval(infer_func, category: Category, dataset_name: str):
    # Create the dataset and dataloader
    if dataset_name == "svt":
        img_dir = "data/svt1_augmented/img/"
        DatasetClass = WeatherAugmentedImageDataset

        xml_path = "data/svt1/test.xml"
        annotations = load_annotations_svt(xml_path)
    elif dataset_name == "svt_cropped":
        img_dir = "data/svt1_augmented/img_cropped/"
        DatasetClass = WeatherAugmentedCroppedImageDataset

        xml_path = "data/svt1/test.xml"
        annotations = load_annotations_svt(xml_path)
    elif dataset_name == "icdar":
        img_dir = "data/icdar2015/augmented/"
        DatasetClass = WeatherAugmentedImageDataset

        xml_path = "data/icdar2015/test_icdar2015_label.txt"
        annotations = load_annotations_icdar(xml_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    dataset = DatasetClass(img_dir, category=category)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process each image with the model
    prompt = "Find and read all text in the image"
    results = []

    correct_ocr = 0
    processed_images = 0

    results = []
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        process_task = progress.add_task("Processing images...", total=len(dataloader))

        for batch in dataloader:
            # Processing code will go here
            progress.update(process_task, advance=1)
            # Get the image path and load image as PIL
            path = batch["path"][0]  # Get the corresponding path
            # print(path)
            # return 1, 2
            # Load the image file directly using PIL - VLM pipelines typically expect PIL Images
            # image_pil = Image.open(path).convert("RGB")

            # Generate response from the model
            # response = pipe(prompt=prompt, images=image_pil, max_new_tokens=512, do_sample=False)
            try:
                result, text_found = infer_func(path)
            except Exception as e:
                logger.error(f"Error processing image {path}: {e}")
                exit()
                continue

            words = [word.strip(".,!?:;()-\"'").lower() for word in text_found.split()]
            augmented_path = result["path"]

            if not "cropped" in dataset_name:
                word_counts = Counter(word for word in words if word)

                logger.info(
                    f"Image: {result['path']}. Word counts: {dict(word_counts)}"
                )

                # Match annotation to each result
                # print(f"augmented_path: {augmented_path}")
                if "svt" in dataset_name:
                    original_filename = get_original_filename_svt(augmented_path)
                elif "icdar" in dataset_name:
                    original_filename = get_original_filename_icdar(augmented_path)
                else:
                    raise ValueError(f"Dataset {dataset_name} not supported")
                # print(annotations)

                if original_filename in annotations:
                    ground_truth = [
                        tag["text"].lower() for tag in annotations[original_filename]
                    ]
                    ground_truth = [
                        word.strip("#.,!?:;()-\"'").lower()
                        for word in ground_truth
                        if word
                    ]
                    word_counts_ground_truth = Counter(
                        word for word in ground_truth if word
                    )

                    logger.info(f"Ground truth: {word_counts_ground_truth}")
                    result["ground_truth"] = ground_truth

                    # Verify that all ground truth words are found in model response
                    all_matched = True
                    missing_words = []
                    mismatched_counts = {}

                    for word, count in word_counts_ground_truth.items():
                        if word not in word_counts:
                            all_matched = False
                            missing_words.append(word)
                        elif word_counts[word] != count:
                            all_matched = False
                            mismatched_counts[word] = (count, word_counts[word])

                    processed_images += 1
                    if all_matched:
                        logger.info(
                            "✅ All ground truth words found with correct counts!"
                        )
                        correct_ocr += 1
                    else:
                        if missing_words:
                            logger.warning(f"❌ Missing words: {missing_words}")
                        if mismatched_counts:
                            logger.warning(
                                f"❌ Mismatched counts: {mismatched_counts} (ground_truth, prediction)"
                            )

                    result["match_success"] = all_matched
                    result["missing_words"] = missing_words
                    result["mismatched_counts"] = mismatched_counts

                else:
                    logger.warning(
                        f"No annotation found for {original_filename} (from {augmented_path})"
                    )
            elif "cropped" in dataset_name:
                # Extract ground truth from filename
                filename = os.path.basename(augmented_path)
                # Remove file extension
                filename_without_ext = os.path.splitext(filename)[0]
                # Split by underscore and get the last part as ground truth
                ground_truth = filename_without_ext.split("_")[-1].lower()

                logger.info(f"Ground truth from filename: {ground_truth}")
                result["ground_truth"] = ground_truth

                # Check if ground truth word is in the predicted words
                if ground_truth in words:
                    logger.info(
                        f"✅ Ground truth '{ground_truth}' found in predictions!"
                    )
                    correct_ocr += 1
                    result["match_success"] = True
                    result["missing_words"] = []
                else:
                    logger.warning(
                        f"❌ Ground truth '{ground_truth}' not found in predictions: {words}"
                    )
                    result["match_success"] = False
                    result["missing_words"] = [ground_truth]

                processed_images += 1
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")

            logger.info(
                f"Processed {processed_images} images. Correct OCR: {correct_ocr}/{processed_images} ({(correct_ocr / processed_images) * 100:.2f}%)"
            )
            results.append(result)

            # Save results to a json file
            with open(
                f"results/{model_name.replace('/', '-')}_{dataset_name}_{category.value}.json",
                "w",
            ) as f:
                json.dump(results, f, indent=4)
    return correct_ocr, processed_images


def infer_func_gemma_qwen(path):
    # Good for Gemma and Qwen
    messages = create_message(path)
    response = pipe(text=messages)
    result = {"path": path, "response": response[0]["generated_text"]}
    text_found = result["response"][-1]["content"]
    return result, text_found


def infer_func_smolvlm(path):
    # # Good for Gemma and Qwen
    messages = create_message(path)
    # response = pipe(text=messages)
    # result = {"path": path, "response": response[0]["generated_text"]}
    # text_found = result["response"][-1]["content"]
    # return result, text_found

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    result = {"path": path, "response": response[0]}
    return result, response[0]


def infer_func_internvl(path):
    # Good for InternVL
    messages = create_message(path)
    message = messages[0]["content"][0]["text"]
    pixel_values = load_image(path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    question = f"<image>\n{message}"
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    result = {"path": path, "response": response}
    return result, response


def infer_func_tinyllava(path):
    messages = create_message(path)
    message = messages[0]["content"][0]["text"]
    response, genertaion_time = model.chat(
        prompt=message, image=path, tokenizer=tokenizer
    )

    result = {"path": path, "response": response}
    return result, response


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate OCR model on SVT dataset.")
    parser.add_argument(
        "--model",
        choices=[
            "google/gemma-3-4b-it",  #
            "Qwen/Qwen2.5-VL-3B-Instruct",  #
            "OpenGVLab/InternVL3-2B",  #
            "OpenGVLab/InternVL3-1B",  #
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",  #
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",  #
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",  #
            "llava-hf/llava-1.5-7b-hf",  #
            "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",  #
            "tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B",  #
            # The ones below are not working
            "jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B",
            "Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP",
            "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP",
            "google/gemma-3n-E4B-it-litert-preview",
        ],
        default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
    )
    parser.add_argument(
        "--dataset",
        choices=["svt", "svt_cropped", "icdar"],
        default="icdar",
    )
    args = parser.parse_args()

    model_name = args.model
    logger.info(f"Model: {model_name}")

    device = "cuda"
    torch_dtype = "bfloat16"

    if model_name in [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "google/gemma-3-4b-it",
        "llava-hf/llava-1.5-7b-hf",
        "google/gemma-3n-E4B-it-litert-preview",
    ]:
        pipe = init_pipeline(model_name, device, torch_dtype)
        infer_func = infer_func_gemma_qwen
    elif model_name in [
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    ]:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        infer_func = infer_func_smolvlm
    elif model_name in [
        "OpenGVLab/InternVL3-2B",
        "OpenGVLab/InternVL3-1B",
    ]:
        from vlm.internvl import device_map, load_image

        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        infer_func = infer_func_internvl
    elif "tinyllava" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.cuda()
        config = model.config
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            model_max_length=config.tokenizer_model_max_length,
            padding_side=config.tokenizer_padding_side,
        )
        infer_func = infer_func_tinyllava
    else:
        raise ValueError(f"Model {model_name} not supported")

    logger.info("Pipeline initialized successfully.")

    category_results = {}
    for category in CATEGORIES:
        logger.info(f"Evaluating category: {category.value}")
        correct_ocr, processed_images = eval(infer_func, category, args.dataset)
        category_results[category.value] = (correct_ocr, processed_images)
        # Save to JSON file
        with open(
            f"results/{model_name.replace('/', '-')}_{args.dataset}_scores.json", "w"
        ) as f:
            json.dump(
                category_results,
                f,
                indent=4,
            )
        # Create a Rich table to display results

        table = Table(title=f"OCR Results for {model_name}")
        table.add_column("Category", style="cyan")
        table.add_column("Correct", style="green")
        table.add_column("Total", style="blue")
        table.add_column("Accuracy (%)", style="yellow")

        for cat, (correct, total) in category_results.items():
            accuracy = (correct / total) * 100 if total > 0 else 0
            table.add_row(cat, str(correct), str(total), f"{accuracy:.2f}")

        console = Console()
        console.print(table)
    logger.info("Evaluation completed.")
