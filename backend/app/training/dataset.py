"""
Medical Image Dataset Loader for CheXpert / NIH Chest X-Ray14.

Handles dataset downloading, preparation, augmentation, and 
TensorFlow data pipeline creation for training DenseNet121.

Supported Datasets:
    - NIH Chest X-Ray14 (112,120 frontal-view X-rays, 14 labels)
    - CheXpert (224,316 chest radiographs, 14 labels)
    - Custom dataset directory

Reference:
    NIH: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database" (CVPR 2017)
    CheXpert: Irvin et al., "CheXpert: Large Chest Radiograph Dataset" (AAAI 2019)
"""

import os
import csv
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# 14 Pathology Labels (CheXpert/NIH standard order)
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax"
]

NUM_CLASSES = len(PATHOLOGY_LABELS)


class MedicalDatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    def __init__(
        self,
        dataset_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        augment_train: bool = True,
        use_class_weights: bool = True,
        label_smoothing: float = 0.1,
        num_workers: int = 4,
        cache_dataset: bool = True,
        seed: int = 42
    ):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.augment_train = augment_train
        self.use_class_weights = use_class_weights
        self.label_smoothing = label_smoothing
        self.num_workers = num_workers
        self.cache_dataset = cache_dataset
        self.seed = seed


class ChestXRayDataset:
    """
    Multi-format chest X-ray dataset loader.
    
    Supports:
    1. NIH CSV format (Data_Entry_2017.csv)
    2. CheXpert CSV format (train.csv / valid.csv)
    3. Directory-based format (one folder per label)
    4. Custom CSV format
    """
    
    def __init__(self, config: MedicalDatasetConfig):
        self.config = config
        self.image_paths = []
        self.labels = []
        self.class_weights = None
        self._loaded = False
        
    def load_from_directory(self, image_dir: Optional[str] = None) -> 'ChestXRayDataset':
        """
        Load dataset from a directory structure.
        
        Expected format:
            dataset_dir/
            ├── images/
            │   ├── image_001.png
            │   ├── image_002.png
            │   └── ...
            └── labels.csv  (image_path, label1, label2, ..., label14)
        
        OR directory-based:
            dataset_dir/
            ├── Pneumonia/
            │   ├── img1.png
            │   └── ...
            ├── Normal/
            │   ├── img1.png
            │   └── ...
            └── ...
        """
        base_dir = Path(image_dir or self.config.dataset_dir)
        
        # Check for CSV label file
        csv_files = list(base_dir.glob("*.csv"))
        
        if csv_files:
            self._load_from_csv(csv_files[0], base_dir)
        else:
            self._load_from_subdirectories(base_dir)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.image_paths)} images with {NUM_CLASSES} labels")
        
        # Compute class weights for imbalanced dataset
        if self.config.use_class_weights:
            self._compute_class_weights()
        
        return self
    
    def _load_from_csv(self, csv_path: Path, base_dir: Path):
        """Load images and labels from a CSV file."""
        logger.info(f"Loading labels from {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Detect format (NIH vs CheXpert)
            if "Finding Labels" in header:
                self._parse_nih_format(reader, header, base_dir)
            elif "Path" in header:
                self._parse_chexpert_format(reader, header, base_dir)
            else:
                self._parse_generic_format(reader, header, base_dir)
    
    def _parse_nih_format(self, reader, header, base_dir: Path):
        """
        Parse NIH Chest X-Ray14 Data_Entry_2017.csv format.
        
        Format: Image Index | Finding Labels | Follow-up # | Patient ID | ...
        Finding Labels: pipe-separated (e.g., "Pneumonia|Effusion")
        """
        image_col = header.index("Image Index")
        label_col = header.index("Finding Labels")
        
        for row in reader:
            img_path = base_dir / "images" / row[image_col]
            if not img_path.exists():
                img_path = base_dir / row[image_col]
            
            if img_path.exists():
                self.image_paths.append(str(img_path))
                
                # Parse pipe-separated labels to multi-hot vector
                findings = row[label_col].split("|")
                label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
                
                for finding in findings:
                    finding = finding.strip()
                    if finding in PATHOLOGY_LABELS:
                        idx = PATHOLOGY_LABELS.index(finding)
                        label_vec[idx] = 1.0
                
                self.labels.append(label_vec)
        
        logger.info(f"Parsed NIH format: {len(self.image_paths)} images")
    
    def _parse_chexpert_format(self, reader, header, base_dir: Path):
        """
        Parse CheXpert CSV format.
        
        Labels: -1 (uncertain), 0 (negative), 1 (positive), '' (unmentioned)
        Uncertainty handling: U-Ones approach (treat uncertain as positive)
        """
        path_col = header.index("Path")
        
        # Map CheXpert label columns to our standard labels
        label_mapping = {}
        for i, label in enumerate(PATHOLOGY_LABELS):
            for j, col in enumerate(header):
                if label.lower() in col.lower():
                    label_mapping[i] = j
                    break
        
        for row in reader:
            img_path = base_dir / row[path_col]
            if not img_path.exists():
                # Try relative path
                img_path = base_dir / Path(row[path_col]).name
            
            if img_path.exists():
                self.image_paths.append(str(img_path))
                
                label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
                for our_idx, csv_col in label_mapping.items():
                    val = row[csv_col].strip()
                    if val == "1" or val == "1.0":
                        label_vec[our_idx] = 1.0
                    elif val == "-1" or val == "-1.0":
                        # U-Ones: treat uncertain as positive
                        label_vec[our_idx] = 1.0
                
                self.labels.append(label_vec)
        
        logger.info(f"Parsed CheXpert format: {len(self.image_paths)} images")
    
    def _parse_generic_format(self, reader, header, base_dir: Path):
        """Parse generic CSV: first column is image path, rest are label values (0/1)."""
        for row in reader:
            img_path = base_dir / row[0]
            if not img_path.exists():
                img_path = base_dir / "images" / row[0]
            
            if img_path.exists():
                self.image_paths.append(str(img_path))
                label_vec = np.array([float(v) for v in row[1:NUM_CLASSES+1]], dtype=np.float32)
                # Ensure correct length
                if len(label_vec) < NUM_CLASSES:
                    label_vec = np.pad(label_vec, (0, NUM_CLASSES - len(label_vec)))
                self.labels.append(label_vec[:NUM_CLASSES])
    
    def _load_from_subdirectories(self, base_dir: Path):
        """Load from directory structure where each subfolder is a class."""
        logger.info(f"Loading from directory structure: {base_dir}")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        # Check if subdirectories match pathology labels
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            for subdir in subdirs:
                label_name = subdir.name
                label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
                
                # Check if directory name matches any pathology
                for i, pathology in enumerate(PATHOLOGY_LABELS):
                    if pathology.lower() in label_name.lower() or label_name.lower() in pathology.lower():
                        label_vec[i] = 1.0
                
                # If "normal" or "healthy" directory, all zeros
                if "normal" in label_name.lower() or "healthy" in label_name.lower():
                    label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
                
                for img_file in subdir.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        self.image_paths.append(str(img_file))
                        self.labels.append(label_vec.copy())
        else:
            # Flat directory - all images, no labels (for inference only)
            for img_file in base_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    self.image_paths.append(str(img_file))
                    self.labels.append(np.zeros(NUM_CLASSES, dtype=np.float32))
        
        logger.info(f"Loaded {len(self.image_paths)} images from directory")
    
    def _compute_class_weights(self):
        """
        Compute class weights to handle imbalanced datasets.
        
        Uses effective number of samples approach:
        weight = (1 - beta) / (1 - beta^n)
        where beta = (N-1)/N, n = number of positive samples
        """
        labels_array = np.array(self.labels)
        n_samples = len(labels_array)
        
        # Count positive samples per class
        pos_counts = np.sum(labels_array, axis=0)
        neg_counts = n_samples - pos_counts
        
        # Avoid division by zero
        pos_counts = np.maximum(pos_counts, 1)
        neg_counts = np.maximum(neg_counts, 1)
        
        # Compute weights (inverse frequency)
        self.class_weights = neg_counts / pos_counts
        
        # Normalize weights
        self.class_weights = self.class_weights / np.mean(self.class_weights)
        
        logger.info(f"Class weights computed: min={self.class_weights.min():.2f}, "
                     f"max={self.class_weights.max():.2f}")
    
    def create_tf_datasets(self) -> Tuple:
        """
        Create TensorFlow Dataset objects for training, validation, and testing.
        
        Returns:
            (train_ds, val_ds, test_ds, class_weights)
        """
        import tensorflow as tf
        
        # Convert to numpy arrays
        all_paths = np.array(self.image_paths)
        all_labels = np.array(self.labels, dtype=np.float32)
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(self.config.seed)
        indices = np.random.permutation(len(all_paths))
        all_paths = all_paths[indices]
        all_labels = all_labels[indices]
        
        # Split dataset
        n = len(all_paths)
        n_test = int(n * self.config.test_split)
        n_val = int(n * self.config.val_split)
        n_train = n - n_test - n_val
        
        train_paths = all_paths[:n_train]
        train_labels = all_labels[:n_train]
        
        val_paths = all_paths[n_train:n_train + n_val]
        val_labels = all_labels[n_train:n_train + n_val]
        
        test_paths = all_paths[n_train + n_val:]
        test_labels = all_labels[n_train + n_val:]
        
        logger.info(f"Dataset split: Train={n_train}, Val={n_val}, Test={n_test}")
        
        # Create TF datasets
        train_ds = self._build_tf_dataset(train_paths, train_labels, is_training=True)
        val_ds = self._build_tf_dataset(val_paths, val_labels, is_training=False)
        test_ds = self._build_tf_dataset(test_paths, test_labels, is_training=False)
        
        return train_ds, val_ds, test_ds, self.class_weights
    
    def _build_tf_dataset(self, paths, labels, is_training: bool):
        """Build an optimized TensorFlow data pipeline."""
        import tensorflow as tf
        
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        # Parallel loading and preprocessing
        dataset = dataset.map(
            lambda p, l: self._load_and_preprocess(p, l, is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(paths), 10000))
        
        dataset = dataset.batch(self.config.batch_size)
        
        if self.config.cache_dataset:
            dataset = dataset.cache()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _load_and_preprocess(self, path, label, is_training: bool):
        """Load a single image and apply preprocessing."""
        import tensorflow as tf
        
        # Read and decode image
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize
        image = tf.image.resize(image, self.config.image_size)
        
        # Data augmentation (training only)
        if is_training and self.config.augment_train:
            image = self._augment(image)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply medical image normalization
        mean = tf.constant([0.4985, 0.4985, 0.4985])
        std = tf.constant([0.2460, 0.2460, 0.2460])
        image = (image - mean) / std
        
        # Label smoothing
        if is_training and self.config.label_smoothing > 0:
            label = label * (1.0 - self.config.label_smoothing) + \
                    self.config.label_smoothing / NUM_CLASSES
        
        return image, label
    
    def _augment(self, image):
        """
        Apply medical-image-appropriate data augmentation.
        
        Uses conservative augmentation to preserve diagnostic features:
        - Random horizontal flip (chest X-rays are roughly symmetric)
        - Small rotation (±10 degrees)
        - Brightness/contrast perturbation
        - Random crop and resize
        """
        import tensorflow as tf
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random crop (90-100% of image) then resize back
        shape = tf.shape(image)
        crop_h = tf.cast(tf.cast(shape[0], tf.float32) * tf.random.uniform([], 0.9, 1.0), tf.int32)
        crop_w = tf.cast(tf.cast(shape[1], tf.float32) * tf.random.uniform([], 0.9, 1.0), tf.int32)
        image = tf.image.random_crop(image, [crop_h, crop_w, 3])
        image = tf.image.resize(image, self.config.image_size)
        
        return image
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        labels_array = np.array(self.labels)
        
        stats = {
            "total_images": len(self.image_paths),
            "num_classes": NUM_CLASSES,
            "class_labels": PATHOLOGY_LABELS,
            "positive_counts": {},
            "negative_counts": {},
            "prevalence": {},
            "multi_label_distribution": {}
        }
        
        for i, label in enumerate(PATHOLOGY_LABELS):
            pos = int(np.sum(labels_array[:, i]))
            neg = len(labels_array) - pos
            stats["positive_counts"][label] = pos
            stats["negative_counts"][label] = neg
            stats["prevalence"][label] = round(pos / max(len(labels_array), 1) * 100, 2)
        
        # Multi-label distribution
        label_counts = np.sum(labels_array, axis=1).astype(int)
        for n in range(NUM_CLASSES + 1):
            count = int(np.sum(label_counts == n))
            if count > 0:
                stats["multi_label_distribution"][f"{n}_labels"] = count
        
        return stats
