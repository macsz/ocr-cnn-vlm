import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from common import logger


def add_rain(
    image,
    rain_drops=800,
    slant=-1,
    drop_length=30,
    drop_width=2,
):
    """Add prominent rain effect to an image."""
    image_rain = image.copy()

    # Set random slant if not provided
    if slant == -1:
        slant = random.randint(-20, 20)

    imshape = image.shape

    # Create rain drops
    for i in range(rain_drops):
        x = random.randint(0, imshape[1] - 1)
        y = random.randint(0, imshape[0] - 1)

        # Draw rain drop line with varying opacity
        opacity = random.randint(150, 230)
        drop_color_random = (opacity, opacity, min(255, opacity + 20))

        # Draw rain drop line
        cv2.line(
            image_rain,
            (x, y),
            (x + slant, y + drop_length),
            drop_color_random,
            drop_width,
        )

    # Add blur to simulate motion
    image_rain = cv2.blur(image_rain, (3, 7))  # Vertical motion blur for rain

    # Blend rain with original image
    result = cv2.addWeighted(image, 0.65, image_rain, 0.35, 0)

    # Add a slight blue tint to simulate rainy atmosphere
    blue_tint = np.zeros_like(result)
    blue_tint[:, :, 0] = 20  # Slight blue tint
    result = cv2.add(result, blue_tint)

    return result


def add_fog(image, fog_coeff=0.8):
    """Add natural cloud-like fog effect to an image."""
    height, width = image.shape[:2]

    # Create base fog layer (grayish-white color for more natural look)
    fog = np.ones((height, width, 3), dtype=np.uint8) * 240

    # Create cloud texture with multiple octaves of noise
    cloud_texture = np.zeros((height, width), dtype=np.float32)

    # Generate multiple layers of noise with different scales
    octaves = 4
    persistence = 0.5

    for octave in range(octaves):
        scale = 2**octave
        weight = persistence**octave

        # Create a noise layer
        noise = np.zeros((height, width), dtype=np.float32)
        temp_noise = np.random.rand(height // scale, width // scale) * 255

        # Resize to full size and normalize
        temp_noise = cv2.resize(temp_noise, (width, height))
        noise = temp_noise.astype(np.float32) / 255.0

        # Add this octave to our cloud texture
        cloud_texture += noise * weight

    # Normalize the result
    cloud_max = np.max(cloud_texture)
    cloud_min = np.min(cloud_texture)
    cloud_texture = (cloud_texture - cloud_min) / (cloud_max - cloud_min)

    # Apply various blurs to create cloud-like formations
    cloud_texture = (cloud_texture * 255).astype(np.uint8)

    # Multilayer blurring for more natural clouds
    small_blur = cv2.GaussianBlur(cloud_texture, (15, 15), 0)
    medium_blur = cv2.GaussianBlur(cloud_texture, (45, 45), 0)
    large_blur = cv2.GaussianBlur(cloud_texture, (85, 85), 0)

    # Combine different blur sizes with varying weights
    cloud_texture = cv2.addWeighted(
        small_blur, 0.3, cv2.addWeighted(medium_blur, 0.5, large_blur, 0.5, 0), 0.7, 0
    )

    # Apply to fog layer with slight variations for RGB channels for realism
    fog[:, :, 0] = cv2.multiply(fog[:, :, 0], cloud_texture, scale=1 / 255)
    fog[:, :, 1] = cv2.multiply(fog[:, :, 1], cloud_texture, scale=1 / 255)
    fog[:, :, 2] = cv2.multiply(fog[:, :, 2], cloud_texture, scale=1 / 255)

    # Add slight color variation (bluish-gray) to simulate natural fog
    fog = fog.astype(np.float32)
    fog[:, :, 0] *= 1.05  # Slightly more blue
    fog[:, :, 1] *= 1.0
    fog[:, :, 2] *= 0.95  # Slightly less red
    fog = np.clip(fog, 0, 255).astype(np.uint8)

    # Apply a gentle gradient from top to bottom (more fog at the top)
    gradient = np.linspace(1.0, 0.7, height).reshape(height, 1, 1)
    fog = (fog * gradient).astype(np.uint8)

    # Blend fog with original image
    foggy_img = cv2.addWeighted(image, 1 - fog_coeff, fog, fog_coeff, 0)

    return foggy_img


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

            # Crop and save tags/labels for each image
            output_dir_cropped = Path("data/svt1_augmented/img_cropped")
            os.makedirs(output_dir_cropped, exist_ok=True)
            if img_path.name in annotations:
                for i, annotation in enumerate(annotations[img_path.name]):
                    tag = annotation["tag"].lower()
                    x, y = annotation["x"], annotation["y"]
                    width, height = annotation["width"], annotation["height"]

                    # Define a safe cropping function
                    def safe_crop(image):
                        try:
                            return image[y : y + height, x : x + width]
                        except IndexError:
                            logger.error(
                                f"Failed to crop {img_path.name} at {x},{y},{width},{height}"
                            )
                            return np.zeros((height, width, 3), dtype=np.uint8)

                    # Crop the tags from each augmentation
                    tag_original = safe_crop(img)
                    tag_rain_light = safe_crop(img_rain_light)
                    tag_rain_medium = safe_crop(img_rain_medium)
                    tag_rain_heavy = safe_crop(img_rain_heavy)
                    tag_fog_light = safe_crop(img_fog_light)
                    tag_fog_medium = safe_crop(img_fog_medium)
                    tag_fog_heavy = safe_crop(img_fog_heavy)
                    tag_rain_fog_light = safe_crop(img_rain_fog_light)
                    tag_rain_fog_medium = safe_crop(img_rain_fog_medium)
                    tag_rain_fog_heavy = safe_crop(img_rain_fog_heavy)

                    # Save cropped tags
                    tag_suffix = f"tag{i}_{tag}" if tag else f"tag{i}"

                    def cv2_imwrite(
                        img,
                        output_dir_cropped,
                        base_name: str,
                        category: str,
                        tag_suffix: str,
                    ):
                        path = str(
                            output_dir_cropped
                            / f"{base_name}_{category}_{tag_suffix}.jpg"
                        )
                        if img is not None and img.size > 0:
                            cv2.imwrite(path, img)
                        else:
                            logger.error(f"Failed to save {path}")

                    cv2_imwrite(
                        tag_original,
                        output_dir_cropped,
                        base_name,
                        "original",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_rain_light,
                        output_dir_cropped,
                        base_name,
                        "rain_light",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_rain_medium,
                        output_dir_cropped,
                        base_name,
                        "rain_medium",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_rain_heavy,
                        output_dir_cropped,
                        base_name,
                        "rain_heavy",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_fog_light,
                        output_dir_cropped,
                        base_name,
                        "fog_light",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_fog_medium,
                        output_dir_cropped,
                        base_name,
                        "fog_medium",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_fog_heavy,
                        output_dir_cropped,
                        base_name,
                        "fog_heavy",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_rain_fog_light,
                        output_dir_cropped,
                        base_name,
                        "rain_fog_light",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_rain_fog_medium,
                        output_dir_cropped,
                        base_name,
                        "rain_fog_medium",
                        tag_suffix,
                    )
                    cv2_imwrite(
                        tag_rain_fog_heavy,
                        output_dir_cropped,
                        base_name,
                        "rain_fog_heavy",
                        tag_suffix,
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


def extract_image_filenames(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_files = []
    for image_elem in root.findall("image"):
        image_name_elem = image_elem.find("imageName")
        if image_name_elem is not None and image_name_elem.text:
            # Get just the filename from the path
            _, filename = os.path.split(image_name_elem.text)
            image_files.append(filename)

    return image_files


def extract_labels_and_locations(xml_path):
    """
    Extract text labels and their bounding box coordinates from an XML file.

    Args:
        xml_path (str): Path to the XML file

    Returns:
        dict: A dictionary where keys are image filenames and values are lists of dictionaries,
                each containing 'tag', 'x', 'y', 'width', 'height' for each tagged rectangle
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {}

    for image_elem in root.findall("image"):
        image_name_elem = image_elem.find("imageName")
        if image_name_elem is not None and image_name_elem.text:
            # Get just the filename from the path
            _, filename = os.path.split(image_name_elem.text)

            # Initialize list for this image
            result[filename] = []

            # Find all tagged rectangles
            tagged_rectangles = image_elem.find("taggedRectangles")
            if tagged_rectangles is not None:
                for rect in tagged_rectangles.findall("taggedRectangle"):
                    # Extract attributes and tag
                    x = int(rect.get("x"))
                    y = int(rect.get("y"))
                    width = int(rect.get("width"))
                    height = int(rect.get("height"))

                    tag_elem = rect.find("tag")
                    tag = tag_elem.text if tag_elem is not None else ""

                    # Add to result
                    result[filename].append(
                        {"tag": tag, "x": x, "y": y, "width": width, "height": height}
                    )

    return result


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    test_xml_path = "data/svt1/test.xml"

    annotations = extract_labels_and_locations(test_xml_path)
    logger.info(f"Found {len(annotations)} images in the XML file")

    input_dir = Path("data/svt1/img")
    output_dir = Path("data/svt1_augmented/img")
    process_images(input_dir, output_dir, annotations)
