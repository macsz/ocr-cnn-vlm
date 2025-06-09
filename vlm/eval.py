import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from enum import Enum
from glob import glob

from rich.console import Console
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import create_message, init_pipeline, logger


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


def get_original_filename(augmented_path):
    """Extract the original filename without augmentation info"""
    # Extract base filename without path and augmentation pattern
    filename = os.path.basename(augmented_path)
    # Extract just the initial part (00_00) from filename like 00_00_rain_light.jpg
    match = re.match(r"(\d+_\d+).*?\.(jpg|png)", filename)
    if match:
        return f"img/{match.group(1)}.jpg"
    return None


def load_annotations(xml_path):
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


def eval(category: Category):
    # Create the dataset and dataloader
    img_dir = "data/svt1_augmented/img"
    dataset = WeatherAugmentedImageDataset(img_dir, category=category)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Process each image with the model
    prompt = "Find and read all text in the image"
    results = []

    # Load annotations
    xml_path = "data/svt1/test.xml"
    annotations = load_annotations(xml_path)

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

            # Load the image file directly using PIL - VLM pipelines typically expect PIL Images
            # image_pil = Image.open(path).convert("RGB")

            # Generate response from the model
            # response = pipe(prompt=prompt, images=image_pil, max_new_tokens=512, do_sample=False)
            messages = create_message(path)
            response = pipe(text=messages)
            result = {"path": path, "response": response[0]["generated_text"]}

            # for result in results:
            text_found = result["response"][-1]["content"]

            words = [word.strip(".,!?:;()-\"'").lower() for word in text_found.split()]
            word_counts = Counter(word for word in words if word)

            logger.info(f"Image: {result['path']}. Word counts: {dict(word_counts)}")

            # Match annotation to each result
            augmented_path = result["path"]
            original_filename = get_original_filename(augmented_path)

            if original_filename in annotations:
                ground_truth = [
                    tag["text"].lower() for tag in annotations[original_filename]
                ]
                ground_truth = [
                    word.strip(".,!?:;()-\"'").lower() for word in ground_truth if word
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
                    logger.info("✅ All ground truth words found with correct counts!")
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

            logger.info(
                f"Processed {processed_images} images. Correct OCR: {correct_ocr}/{processed_images} ({(correct_ocr / processed_images) * 100:.2f}%)"
            )
            results.append(result)

            # Save results to a json file
            with open(
                f"results/{model_name.replace('/', '-')}_{category.value}.json", "w"
            ) as f:
                json.dump(results, f, indent=4)
    return correct_ocr, processed_images


if __name__ == "__main__":

    model_name = "google/gemma-3-4b-it"
    device = "cuda"
    torch_dtype = "bfloat16"

    pipe = init_pipeline(model_name, device, torch_dtype)

    logger.info("Pipeline initialized successfully.")

    categories = [
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
    category_results = {}
    for category in categories:
        logger.info(f"Evaluating category: {category.value}")
        correct_ocr, processed_images = eval(category)
        category_results[category.value] = (correct_ocr, processed_images)
        # Save to JSON file
        with open(f"results/{model_name.replace('/', '-')}_scores.json", "w") as f:
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
