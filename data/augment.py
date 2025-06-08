import os
import random
from pathlib import Path

import cv2
import numpy as np
import tqdm


def add_rain(
    image,
    rain_drops=800,
    slant=-1,
    drop_length=30,
    drop_width=2,
    drop_color=(180, 180, 200),
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


def process_images(input_dir: Path, output_dir: Path):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    # Augment each image
    for img_path in tqdm.tqdm(image_files):
        # Read image
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Failed to read {img_path}")
            continue

        # Process images with different effects
        img_rain = add_rain(img)
        img_fog = add_fog(img)
        img_rain_fog = add_fog(add_rain(img))

        # Save processed images
        base_name = img_path.stem
        cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), img)
        cv2.imwrite(str(output_dir / f"{base_name}_rain.jpg"), img_rain)
        cv2.imwrite(str(output_dir / f"{base_name}_fog.jpg"), img_fog)
        cv2.imwrite(str(output_dir / f"{base_name}_rain_fog.jpg"), img_rain_fog)


if __name__ == "__main__":
    input_dir = Path("data/svt1/img")
    output_dir = Path("data/svt1_augmented/img")
    process_images(input_dir, output_dir)
