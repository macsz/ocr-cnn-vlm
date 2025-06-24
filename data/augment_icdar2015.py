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
ANNOTATIONS_PATH = "/project/ml_datasets/user_datasets/lihang.ying/icdar2015/text_localization/test_icdar2015_label.txt"


def process_images(
    input_dir: Path, output_dir: Path, annotations: Optional[dict[str, list]] = None
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    # Filter by specific filenames if provided
    if annotations is not None:
        image_files = [f for f in image_files if f.name in annotations.keys()]
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

            # Save processed images
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
    process_images(
        input_dir=Path(DATASET_PATH), output_dir=Path("icdar2015/augmented/")
    )
