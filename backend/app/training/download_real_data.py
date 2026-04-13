"""
Download Real Chest X-Ray Dataset via MedMNIST.

Downloads the ChestMNIST dataset — real NIH Chest X-Ray14 images with
all 14 pathology labels. Converts to image files + CSV for our training pipeline.

Dataset source: NIH Clinical Center (via MedMNIST)
    - 112,120 frontal-view X-ray images
    - 14 binary disease labels
    - Pre-split into train/val/test

Usage:
    python -m app.training.download_real_data --output_dir ./real_data --max_images 3000
    python -m app.training.download_real_data --output_dir ./real_data --max_images 5000 --size 64
"""

import os
import sys
import csv
import logging
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 14 Pathology Labels (NIH standard order as used in MedMNIST)
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax"
]


def download_chest_xray_dataset(
    output_dir: str = "./real_data",
    max_images: int = 3000,
    image_size: int = 28,
    include_val: bool = True,
    include_test: bool = True
):
    """
    Download ChestMNIST (real NIH Chest X-Ray14 images) and convert to
    our training format (images/ folder + labels.csv).
    
    Args:
        output_dir: Output directory for the dataset
        max_images: Maximum number of training images to use (for faster training)
        image_size: MedMNIST image size (28, 64, 128, or 224)
        include_val: Include validation split images  
        include_test: Include test split images
    """
    import medmnist
    from medmnist import ChestMNIST
    
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  AI MedScan — Real Dataset Download")
    logger.info("  Source: NIH Chest X-Ray14 (via MedMNIST)")
    logger.info("=" * 60)
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Max training images: {max_images}")
    logger.info(f"  MedMNIST image size: {image_size}x{image_size}")
    logger.info("=" * 60)
    
    # ── Download ChestMNIST ──
    logger.info("\n📥 Downloading ChestMNIST dataset...")
    logger.info("   (This downloads real NIH Chest X-Ray14 images)")
    
    download_root = str(output_path.resolve() / ".medmnist_cache")
    os.makedirs(download_root, exist_ok=True)
    
    # Download all splits
    train_dataset = ChestMNIST(
        split="train",
        download=True,
        root=download_root,
        size=image_size
    )
    
    logger.info(f"   ✅ Training set: {len(train_dataset)} images")
    
    val_dataset = None
    test_dataset = None
    
    if include_val:
        val_dataset = ChestMNIST(
            split="val",
            download=True,
            root=download_root,
            size=image_size
        )
        logger.info(f"   ✅ Validation set: {len(val_dataset)} images")
    
    if include_test:
        test_dataset = ChestMNIST(
            split="test",
            download=True, 
            root=download_root,
            size=image_size
        )
        logger.info(f"   ✅ Test set: {len(test_dataset)} images")
    
    # ── Convert to our format ──
    logger.info("\n🔄 Converting to AI MedScan training format...")
    
    csv_lines = ["image_path," + ",".join(PATHOLOGY_LABELS)]
    total_saved = 0
    label_counts = np.zeros(14)
    
    # Process training data (with limit)
    train_limit = min(max_images, len(train_dataset))
    
    # Shuffle indices for representative sampling
    np.random.seed(42)
    train_indices = np.random.permutation(len(train_dataset))[:train_limit]
    
    logger.info(f"\n   Processing {train_limit} training images...")
    for idx in tqdm(train_indices, desc="   Saving train images", ncols=80):
        img, label = train_dataset[idx]
        
        # MedMNIST returns PIL Image and numpy label
        if not isinstance(img, Image.Image):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            else:
                # It's a torch tensor
                img = Image.fromarray(img.numpy().squeeze())
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 for DenseNet121
        img_resized = img.resize((224, 224), Image.LANCZOS)
        
        filename = f"nih_train_{total_saved:06d}.png"
        img_resized.save(images_dir / filename, optimize=True)
        
        # Convert label to our format
        label_vec = np.zeros(14, dtype=int)
        if isinstance(label, np.ndarray):
            label_array = label.flatten()
        else:
            label_array = np.array(label).flatten()
        
        for i in range(min(len(label_array), 14)):
            label_vec[i] = int(label_array[i])
        
        label_counts += label_vec
        label_str = ",".join([str(v) for v in label_vec])
        csv_lines.append(f"{filename},{label_str}")
        total_saved += 1
    
    # Process validation data
    if val_dataset:
        val_limit = min(max_images // 5, len(val_dataset))
        val_indices = np.random.permutation(len(val_dataset))[:val_limit]
        
        logger.info(f"   Processing {val_limit} validation images...")
        for idx in tqdm(val_indices, desc="   Saving val images", ncols=80):
            img, label = val_dataset[idx]
            
            if not isinstance(img, Image.Image):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                else:
                    img = Image.fromarray(img.numpy().squeeze())
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize((224, 224), Image.LANCZOS)
            
            filename = f"nih_val_{total_saved:06d}.png"
            img_resized.save(images_dir / filename, optimize=True)
            
            label_vec = np.zeros(14, dtype=int)
            if isinstance(label, np.ndarray):
                label_array = label.flatten()
            else:
                label_array = np.array(label).flatten()
            
            for i in range(min(len(label_array), 14)):
                label_vec[i] = int(label_array[i])
            
            label_counts += label_vec
            label_str = ",".join([str(v) for v in label_vec])
            csv_lines.append(f"{filename},{label_str}")
            total_saved += 1
    
    # Process test data
    if test_dataset:
        test_limit = min(max_images // 5, len(test_dataset))
        test_indices = np.random.permutation(len(test_dataset))[:test_limit]
        
        logger.info(f"   Processing {test_limit} test images...")
        for idx in tqdm(test_indices, desc="   Saving test images", ncols=80):
            img, label = test_dataset[idx]
            
            if not isinstance(img, Image.Image):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                else:
                    img = Image.fromarray(img.numpy().squeeze())
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize((224, 224), Image.LANCZOS)
            
            filename = f"nih_test_{total_saved:06d}.png"
            img_resized.save(images_dir / filename, optimize=True)
            
            label_vec = np.zeros(14, dtype=int)
            if isinstance(label, np.ndarray):
                label_array = label.flatten()
            else:
                label_array = np.array(label).flatten()
            
            for i in range(min(len(label_array), 14)):
                label_vec[i] = int(label_array[i])
            
            label_counts += label_vec
            label_str = ",".join([str(v) for v in label_vec])
            csv_lines.append(f"{filename},{label_str}")
            total_saved += 1
    
    # ── Write labels CSV ──
    csv_path = output_path / "labels.csv"
    with open(csv_path, 'w', newline='') as f:
        f.write("\n".join(csv_lines))
    
    # ── Write dataset info ──
    import json
    info = {
        "dataset_type": "real_nih_chest_xray",
        "source": "NIH Chest X-Ray14 (via MedMNIST)",
        "total_images": total_saved,
        "image_size": "224x224 (resized from {}x{})".format(image_size, image_size),
        "num_classes": 14,
        "class_labels": PATHOLOGY_LABELS,
        "label_distribution": {
            label: int(count) for label, count in zip(PATHOLOGY_LABELS, label_counts)
        },
        "splits": {
            "train": train_limit,
            "val": val_limit if val_dataset else 0,
            "test": test_limit if test_dataset else 0
        }
    }
    
    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # ── Print summary ──
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ✅ Real Dataset Downloaded Successfully!")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total images: {total_saved}")
    logger.info(f"  Images dir:   {images_dir}")
    logger.info(f"  Labels CSV:   {csv_path}")
    logger.info(f"\n  Label Distribution:")
    for label, count in zip(PATHOLOGY_LABELS, label_counts):
        pct = count / total_saved * 100
        bar = "█" * int(pct / 2)
        logger.info(f"    {label:>20s}: {int(count):>5d} ({pct:>5.1f}%) |{bar}")
    
    no_finding = total_saved - int(np.sum(np.any(
        np.array([list(map(int, line.split(',')[1:])) for line in csv_lines[1:]]), axis=1
    )))
    logger.info(f"    {'No Finding':>20s}: {no_finding:>5d} ({no_finding/total_saved*100:>5.1f}%)")
    
    logger.info(f"\n  📋 Next step — Train on real data:")
    logger.info(f"     python -m app.training.train_model --data_dir ./real_data --epochs 15 --batch_size 16")
    logger.info(f"{'=' * 60}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="AI MedScan — Download Real NIH Chest X-Ray Dataset"
    )
    parser.add_argument('--output_dir', type=str, default='./real_data',
                        help='Output directory')
    parser.add_argument('--max_images', type=int, default=3000,
                        help='Maximum training images (more = better accuracy but slower)')
    parser.add_argument('--size', type=int, default=28, choices=[28, 64, 128, 224],
                        help='MedMNIST download resolution (28=fastest, 224=best quality)')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    
    download_chest_xray_dataset(
        output_dir=args.output_dir,
        max_images=args.max_images,
        image_size=args.size
    )


if __name__ == "__main__":
    main()
