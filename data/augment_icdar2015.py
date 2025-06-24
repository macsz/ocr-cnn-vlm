import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from common import logger
from data import add_fog, add_rain

DATASET_PATH = "/project/ml_datasets/user_datasets/lihang.ying/icdar2015/text_localization/ch4_test_images"
ANNOTATIONS_PATH = "data/icdar2015/test_icdar2015_label.txt"


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
                transcription = transcription.strip("@#.,!?:;()-\"'").lower()

                tags.append({"text": transcription, "bbox": ann["points"]})

            annotations[image_path] = tags

    return annotations


def crop_and_warp_polygon(img, points):
    """
    Warp a quadrilateral region of an image into a straight rectangle.
    Args:
        img: Loaded image (BGR)
        points: List of 4 (x, y) points in polygon order
    Returns:
        Cropped and warped image region
    """
    pts = np.array(points, dtype="float32")

    # Compute width and height of the destination rectangle
    def compute_box_dims(pts):
        width = max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3])
        )
        height = max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2])
        )
        return int(width), int(height)

    w, h = compute_box_dims(pts)

    dst_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))

    h, w = warped.shape[:2]
    min_dim = min(h, w)

    if min_dim < 28:
        # Calculate scale factor to make minimum dimension 28
        scale = 28.0 / min_dim
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize using cv2.INTER_LINEAR for upscaling
        warped = cv2.resize(
            warped,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return warped


def process_images(
    input_dir: Path, output_dir: Path, annotations: Optional[dict[str, list]] = None
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    cropped_output_dir = Path("data/icdar2015/cropped/")
    os.makedirs(cropped_output_dir, exist_ok=True)

    # Get all image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    logger.info(f"Found {len(image_files)} images to process in {input_dir}")

    # Augment each image
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for img_path in image_files:
            ann = annotations["ch4_test_images/" + str(img_path.name)]
            logger.info(f"Processing {img_path} with annotations {ann}")
            progress.update(task, advance=1)
            base_name = img_path.stem
            # Read image
            img = cv2.imread(str(img_path))

            if img is None:
                logger.error(f"Failed to read {img_path}")
                continue

            # Process images with different effects
            coef_rain_light, coef_rain_medium, coef_rain_heavy = (
                int(800 * 1.0),
                int(800 * 2.0),
                int(800 * 3.0),
            )
            coef_fog_light, coef_fog_medium, coef_fog_heavy = 0.5, 0.7, 0.9

            img_rain_light = add_rain(img, rain_drops=coef_rain_light)
            img_rain_medium = add_rain(img, rain_drops=coef_rain_medium)
            img_rain_heavy = add_rain(img, rain_drops=coef_rain_heavy)

            img_fog_light = add_fog(img, fog_coeff=coef_fog_light)
            img_fog_medium = add_fog(img, fog_coeff=coef_fog_medium)
            img_fog_heavy = add_fog(img, fog_coeff=coef_fog_heavy)

            img_rain_fog_light = add_fog(
                add_rain(img, rain_drops=coef_rain_light), fog_coeff=coef_fog_light
            )
            img_rain_fog_medium = add_fog(
                add_rain(img, rain_drops=coef_rain_medium), fog_coeff=coef_fog_medium
            )
            img_rain_fog_heavy = add_fog(
                add_rain(img, rain_drops=coef_rain_heavy), fog_coeff=coef_fog_heavy
            )

            os.makedirs(cropped_output_dir, exist_ok=True)
            for i, tag in enumerate(ann):
                category = "original"
                cropped_image = crop_and_warp_polygon(img, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "rain_light"
                cropped_image = crop_and_warp_polygon(img_rain_light, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "rain_medium"
                cropped_image = crop_and_warp_polygon(img_rain_medium, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "rain_heavy"
                cropped_image = crop_and_warp_polygon(img_rain_heavy, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "fog_light"
                cropped_image = crop_and_warp_polygon(img_fog_light, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "fog_medium"
                cropped_image = crop_and_warp_polygon(img_fog_medium, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "fog_heavy"
                cropped_image = crop_and_warp_polygon(img_fog_heavy, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "rain_fog_light"
                cropped_image = crop_and_warp_polygon(img_rain_fog_light, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "rain_fog_medium"
                cropped_image = crop_and_warp_polygon(img_rain_fog_medium, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                category = "rain_fog_heavy"
                cropped_image = crop_and_warp_polygon(img_rain_fog_heavy, tag["bbox"])
                cv2.imwrite(str(cropped_output_dir / f"{base_name}_{category}_tag{i}_{tag['text']}.jpg"), cropped_image)

                # return
            # Save processed images
            # continue
            cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), img)
            cv2.imwrite(str(output_dir / f"{base_name}_rain_light.jpg"), img_rain_light)
            cv2.imwrite(
                str(output_dir / f"{base_name}_rain_medium.jpg"), img_rain_medium
            )
            cv2.imwrite(str(output_dir / f"{base_name}_rain_heavy.jpg"), img_rain_heavy)

            cv2.imwrite(str(output_dir / f"{base_name}_fog_light.jpg"), img_fog_light)
            cv2.imwrite(str(output_dir / f"{base_name}_fog_medium.jpg"), img_fog_medium)
            cv2.imwrite(str(output_dir / f"{base_name}_fog_heavy.jpg"), img_fog_heavy)

            cv2.imwrite(
                str(output_dir / f"{base_name}_rain_fog_light.jpg"), img_rain_fog_light
            )
            cv2.imwrite(
                str(output_dir / f"{base_name}_rain_fog_medium.jpg"),
                img_rain_fog_medium,
            )
            cv2.imwrite(
                str(output_dir / f"{base_name}_rain_fog_heavy.jpg"), img_rain_fog_heavy
            )

            # Save all images in a single image in 2x2 grid
            combined_img = np.zeros(
                (img.shape[0] * 2, img.shape[1] * 2, 3), dtype=np.uint8
            )
            combined_img[: img.shape[0], : img.shape[1]] = img
            combined_img[: img.shape[0], img.shape[1] :] = img_rain_medium
            combined_img[img.shape[0] :, : img.shape[1]] = img_fog_medium
            combined_img[img.shape[0] :, img.shape[1] :] = img_rain_fog_medium
            cv2.imwrite(str(output_dir / f"{base_name}_combined.jpg"), combined_img)

            # Create a 3x3 grid showing all variations of rain/fog/combined with light/medium/heavy intensities
            grid_images = [
                img_rain_light,
                img_rain_medium,
                img_rain_heavy,
                img_fog_light,
                img_fog_medium,
                img_fog_heavy,
                img_rain_fog_light,
                img_rain_fog_medium,
                img_rain_fog_heavy,
            ]

            # Calculate grid dimensions
            cell_height, cell_width = img.shape[0], img.shape[1]
            grid_img = np.zeros((cell_height * 3, cell_width * 3, 3), dtype=np.uint8)

            # Fill the grid with images
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    y_start = i * cell_height
                    y_end = (i + 1) * cell_height
                    x_start = j * cell_width
                    x_end = (j + 1) * cell_width
                    grid_img[y_start:y_end, x_start:x_end] = grid_images[idx]

            # Save the 3x3 grid
            cv2.imwrite(str(output_dir / f"{base_name}_combined_3x3.jpg"), grid_img)


if __name__ == "__main__":
    annotations = load_annotations_icdar(ANNOTATIONS_PATH)
    process_images(
        input_dir=Path(DATASET_PATH),
        output_dir=Path("data/icdar2015/augmented/"),
        annotations=annotations,
    