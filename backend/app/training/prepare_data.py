"""
Dataset Download & Preparation Utility.

Downloads and prepares popular chest X-ray datasets for training:
    - NIH Chest X-Ray14 (small subset for demo)
    - Generates synthetic training data for quick testing

For full datasets, provides instructions and links.

Usage:
    python -m app.training.prepare_data --output_dir ./data --mode demo
"""

import os
import sys
import json
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


DATASET_INFO = {
    "nih_chest_xray14": {
        "name": "NIH Chest X-Ray14",
        "description": "112,120 frontal-view X-ray images with 14 disease labels",
        "url": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "paper": "Wang et al., ChestX-ray8 (CVPR 2017)",
        "size": "~42 GB",
        "labels": 14,
        "images": 112120
    },
    "chexpert": {
        "name": "CheXpert",
        "description": "224,316 chest radiographs with uncertainty labels",
        "url": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "paper": "Irvin et al. (AAAI 2019)",
        "size": "~439 GB (full) / ~11 GB (small)",
        "labels": 14,
        "images": 224316
    },
    "rsna_pneumonia": {
        "name": "RSNA Pneumonia Detection",
        "description": "Stage 2 pneumonia detection challenge dataset",
        "url": "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge",
        "paper": "RSNA 2018",
        "size": "~3.5 GB",
        "labels": 1,
        "images": 26684
    }
}


def generate_demo_dataset(output_dir: str, num_images: int = 200, image_size: int = 224):
    """
    Generate a synthetic demo dataset for quick testing.
    
    Creates realistic-looking synthetic chest X-ray-like images
    with random multi-label annotations for testing the training pipeline.
    """
    from PIL import Image, ImageDraw, ImageFilter
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    from app.training.dataset import PATHOLOGY_LABELS
    
    logger.info(f"Generating {num_images} synthetic training images...")
    
    # Create labels CSV
    csv_lines = ["image_path," + ",".join(PATHOLOGY_LABELS)]
    
    for i in range(num_images):
        # Generate synthetic chest X-ray-like image
        img = generate_synthetic_xray(image_size)
        
        filename = f"synth_xray_{i:05d}.png"
        img.save(images_dir / filename)
        
        # Generate random multi-label annotations
        # Simulate realistic label distribution
        labels = generate_realistic_labels()
        label_str = ",".join([str(int(l)) for l in labels])
        csv_lines.append(f"{filename},{label_str}")
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Generated {i + 1}/{num_images} images")
    
    # Write labels CSV
    csv_path = output_path / "labels.csv"
    with open(csv_path, 'w') as f:
        f.write("\n".join(csv_lines))
    
    # Write dataset info
    info = {
        "dataset_type": "synthetic_demo",
        "num_images": num_images,
        "image_size": image_size,
        "num_classes": len(PATHOLOGY_LABELS),
        "class_labels": PATHOLOGY_LABELS,
        "note": "This is a synthetic dataset for testing the training pipeline. "
                "For real results, use NIH Chest X-Ray14 or CheXpert."
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"\n✅ Demo dataset generated:")
    logger.info(f"   Images: {images_dir} ({num_images} files)")
    logger.info(f"   Labels: {csv_path}")
    logger.info(f"   Info:   {output_path / 'dataset_info.json'}")
    
    return str(output_path)


def generate_synthetic_xray(size: int = 224) -> 'Image':
    """
    Generate a synthetic chest X-ray-like image.
    
    Creates a grayscale image with structures resembling
    chest X-ray features (ribcage outline, lung fields, cardiac shadow).
    """
    from PIL import Image, ImageDraw, ImageFilter
    
    img = Image.new('L', (size, size), color=20)
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size // 2, size // 2
    
    # Background gradient (darker at edges)
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            gradient = int(30 + 50 * (1 - dist / max_dist))
            if img.getpixel((x, y)) < gradient:
                img.putpixel((x, y), gradient)
    
    # Lung fields (brighter elliptical areas)
    # Right lung
    for angle in range(360):
        rad = np.radians(angle)
        rx, ry = size * 0.18, size * 0.28
        cx, cy = center_x - size * 0.15, center_y - size * 0.02
        for r_scale in np.linspace(0, 1, 20):
            px = int(cx + r_scale * rx * np.cos(rad))
            py = int(cy + r_scale * ry * np.sin(rad))
            if 0 <= px < size and 0 <= py < size:
                val = int(80 + 60 * (1 - r_scale) + np.random.normal(0, 5))
                img.putpixel((px, py), max(0, min(255, val)))
    
    # Left lung
    for angle in range(360):
        rad = np.radians(angle)
        rx, ry = size * 0.18, size * 0.28
        cx, cy = center_x + size * 0.15, center_y - size * 0.02
        for r_scale in np.linspace(0, 1, 20):
            px = int(cx + r_scale * rx * np.cos(rad))
            py = int(cy + r_scale * ry * np.sin(rad))
            if 0 <= px < size and 0 <= py < size:
                val = int(80 + 60 * (1 - r_scale) + np.random.normal(0, 5))
                img.putpixel((px, py), max(0, min(255, val)))
    
    # Cardiac shadow (darker central area)
    for angle in range(360):
        rad = np.radians(angle)
        rx, ry = size * 0.12, size * 0.15
        cx, cy = center_x - size * 0.03, center_y + size * 0.05
        for r_scale in np.linspace(0, 1, 15):
            px = int(cx + r_scale * rx * np.cos(rad))
            py = int(cy + r_scale * ry * np.sin(rad))
            if 0 <= px < size and 0 <= py < size:
                val = int(40 + 20 * r_scale)
                img.putpixel((px, py), max(0, min(255, val)))
    
    # Ribs (horizontal curved lines)
    for rib in range(6):
        y_base = int(center_y - size * 0.25 + rib * size * 0.09)
        for x in range(int(size * 0.15), int(size * 0.85)):
            curve = int(5 * np.sin((x - size * 0.15) / (size * 0.7) * np.pi))
            y = y_base + curve + np.random.randint(-1, 2)
            if 0 <= y < size:
                val = img.getpixel((x, y))
                img.putpixel((x, y), max(0, min(255, val + 30)))
    
    # Add noise
    noise = np.random.normal(0, 8, (size, size)).astype(np.int16)
    img_array = np.array(img).astype(np.int16) + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    
    # Gaussian blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Random pathological features
    if np.random.random() < 0.3:
        # Add opacity (bright spot) simulating pathology
        draw = ImageDraw.Draw(img)
        px = np.random.randint(size * 0.2, size * 0.8)
        py = np.random.randint(size * 0.2, size * 0.8)
        r = np.random.randint(10, 30)
        for angle in range(360):
            rad = np.radians(angle)
            for rs in np.linspace(0, 1, 10):
                x = int(px + rs * r * np.cos(rad))
                y = int(py + rs * r * np.sin(rad))
                if 0 <= x < size and 0 <= y < size:
                    val = img.getpixel((x, y))
                    opacity = int(40 * (1 - rs))
                    img.putpixel((x, y), min(255, val + opacity))
    
    # Convert to RGB (model expects 3 channels)
    img_rgb = Image.merge('RGB', [img, img, img])
    
    return img_rgb


def generate_realistic_labels() -> np.ndarray:
    """
    Generate realistic multi-label distributions.
    
    Mimics the actual prevalence of pathologies in CheXpert dataset.
    """
    from app.training.dataset import NUM_CLASSES
    
    # Approximate CheXpert prevalence rates
    prevalence = [
        0.103,  # Atelectasis
        0.026,  # Cardiomegaly
        0.036,  # Consolidation
        0.058,  # Edema
        0.127,  # Effusion
        0.022,  # Emphysema
        0.015,  # Fibrosis
        0.002,  # Hernia
        0.097,  # Infiltration
        0.051,  # Mass
        0.056,  # Nodule
        0.030,  # Pleural Thickening
        0.012,  # Pneumonia
        0.047   # Pneumothorax
    ]
    
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)
    
    # 40% chance of "no finding" (all zeros)
    if np.random.random() < 0.4:
        return labels
    
    # Generate labels based on prevalence
    for i, prev in enumerate(prevalence):
        if np.random.random() < prev * 5:  # Amplify for demo
            labels[i] = 1.0
    
    # Add correlated findings (e.g., pneumonia often co-occurs with consolidation)
    if labels[12] == 1:  # Pneumonia
        if np.random.random() < 0.5:
            labels[2] = 1  # Consolidation
        if np.random.random() < 0.3:
            labels[8] = 1  # Infiltration
    
    if labels[3] == 1:  # Edema
        if np.random.random() < 0.4:
            labels[4] = 1  # Effusion
    
    return labels


def print_dataset_info():
    """Print information about available datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("  Available Medical Image Datasets")
    logger.info("=" * 70)
    
    for key, info in DATASET_INFO.items():
        logger.info(f"\n  📦 {info['name']}")
        logger.info(f"     {info['description']}")
        logger.info(f"     Images: {info['images']:,} | Labels: {info['labels']}")
        logger.info(f"     Size: {info['size']}")
        logger.info(f"     URL: {info['url']}")
        logger.info(f"     Paper: {info['paper']}")
    
    logger.info(f"\n{'='*70}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AI MedScan — Dataset Preparation")
    
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for dataset')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'info'],
                        help='Mode: demo (generate synthetic), info (show dataset links)')
    parser.add_argument('--num_images', type=int, default=200,
                        help='Number of synthetic images to generate (demo mode)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for generated images')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    if args.mode == 'info':
        print_dataset_info()
    elif args.mode == 'demo':
        generate_demo_dataset(
            output_dir=args.output_dir,
            num_images=args.num_images,
            image_size=args.image_size
        )
        
        logger.info("\n📋 Next steps:")
        logger.info("  1. Generate demo data:")
        logger.info("     python -m app.training.prepare_data --mode demo --num_images 500")
        logger.info("  2. Train model:")
        logger.info("     python -m app.training.train_model --data_dir ./data --epochs 10")
        logger.info("  3. Evaluate:")
        logger.info("     python -m app.training.evaluate --model_path ./trained_models/best_model.h5 --data_dir ./data")


if __name__ == "__main__":
    main()
