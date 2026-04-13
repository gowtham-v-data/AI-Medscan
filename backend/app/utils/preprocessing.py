"""
Image Preprocessing Pipeline for Medical Image Analysis.

This module handles all image preprocessing operations required
for the DenseNet121-based medical image classifier, including
CLAHE enhancement, normalization, and augmentation.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import cv2
from typing import Tuple, Optional


class MedicalImagePreprocessor:
    """
    Industry-grade medical image preprocessing pipeline.
    
    Implements CLAHE (Contrast Limited Adaptive Histogram Equalization),
    normalization, and quality assessment for medical imaging inputs.
    """
    
    # Standard input size for DenseNet121
    TARGET_SIZE = (224, 224)
    
    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    # Medical image specific constants
    CHEST_XRAY_MEAN = np.array([0.4985, 0.4985, 0.4985])
    CHEST_XRAY_STD = np.array([0.2460, 0.2460, 0.2460])
    
    def __init__(self, use_medical_normalization: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_medical_normalization: If True, uses chest X-ray specific
                normalization constants instead of ImageNet defaults.
        """
        self.use_medical_norm = use_medical_normalization
        self.mean = self.CHEST_XRAY_MEAN if use_medical_normalization else self.IMAGENET_MEAN
        self.std = self.CHEST_XRAY_STD if use_medical_normalization else self.IMAGENET_STD
    
    def load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes and convert to numpy array."""
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert grayscale to RGB (common for X-rays)
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0,
                    tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        This is critical for medical images where contrast variations
        can significantly impact diagnostic accuracy.
        
        Args:
            image: Input image as numpy array (RGB)
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space for better CLAHE application
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L-channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def resize_image(self, image: np.ndarray, 
                     target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        Uses center-crop strategy for medical images to preserve
        the region of interest.
        """
        if target_size is None:
            target_size = self.TARGET_SIZE
        
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scale to fit the shorter dimension
        scale = max(target_h / h, target_w / w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop
        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2
        
        cropped = resized[start_h:start_h + target_h, start_w:start_w + target_w]
        return cropped
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values using appropriate mean/std.
        
        Args:
            image: Input image with pixel values in [0, 255]
            
        Returns:
            Normalized image with pixel values centered around 0
        """
        # Scale to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply mean/std normalization
        normalized = (normalized - self.mean) / self.std
        
        return normalized
    
    def assess_quality(self, image: np.ndarray) -> dict:
        """
        Assess the quality of the input medical image.
        
        Returns metrics including:
        - brightness: Average pixel intensity
        - contrast: Standard deviation of pixel values
        - sharpness: Laplacian variance (focus measure)
        - noise_level: Estimated noise level
        - quality_score: Overall quality score (0-100)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Brightness
        brightness = float(np.mean(gray))
        
        # Contrast
        contrast = float(np.std(gray))
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        # Noise estimation (using median absolute deviation)
        sigma = float(np.median(np.abs(gray.astype(float) - np.median(gray))) * 1.4826)
        
        # Overall quality score
        brightness_score = 100 - abs(brightness - 127.5) / 1.275  # Optimal around 127.5
        contrast_score = min(contrast / 0.6, 100)
        sharpness_score = min(sharpness / 5.0, 100)
        noise_score = max(0, 100 - sigma * 2)
        
        quality_score = (
            brightness_score * 0.2 +
            contrast_score * 0.3 +
            sharpness_score * 0.3 +
            noise_score * 0.2
        )
        
        return {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "sharpness": round(sharpness, 2),
            "noise_level": round(sigma, 2),
            "quality_score": round(min(max(quality_score, 0), 100), 1),
            "quality_grade": self._get_quality_grade(quality_score),
            "recommendations": self._get_quality_recommendations(
                brightness, contrast, sharpness, sigma
            )
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Acceptable"
        elif score >= 40:
            return "Poor"
        else:
            return "Unacceptable"
    
    def _get_quality_recommendations(
        self, brightness: float, contrast: float,
        sharpness: float, noise: float
    ) -> list:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if brightness < 50:
            recommendations.append("Image appears too dark. Consider adjusting exposure.")
        elif brightness > 200:
            recommendations.append("Image appears overexposed. Consider reducing brightness.")
        
        if contrast < 30:
            recommendations.append("Low contrast detected. CLAHE enhancement recommended.")
        
        if sharpness < 100:
            recommendations.append("Image may be blurry. Ensure proper focus during acquisition.")
        
        if noise > 30:
            recommendations.append("High noise level detected. Consider noise reduction.")
        
        if not recommendations:
            recommendations.append("Image quality meets diagnostic standards.")
        
        return recommendations
    
    def preprocess(self, image_bytes: bytes, apply_enhancement: bool = True) -> dict:
        """
        Full preprocessing pipeline.
        
        Args:
            image_bytes: Raw image bytes
            apply_enhancement: Whether to apply CLAHE enhancement
            
        Returns:
            Dictionary containing:
            - processed_image: Preprocessed numpy array ready for model input
            - original_image: Original image as numpy array
            - quality_metrics: Image quality assessment
            - metadata: Processing metadata
        """
        # Load image
        original = self.load_image(image_bytes)
        original_size = original.shape[:2]
        
        # Quality assessment on original
        quality_metrics = self.assess_quality(original)
        
        # Apply CLAHE if requested
        if apply_enhancement:
            enhanced = self.apply_clahe(original)
        else:
            enhanced = original.copy()
        
        # Resize
        resized = self.resize_image(enhanced)
        
        # Normalize for model input
        normalized = self.normalize(resized)
        
        # Add batch dimension
        model_input = np.expand_dims(normalized, axis=0)
        
        return {
            "model_input": model_input,
            "original_image": original,
            "enhanced_image": enhanced,
            "resized_image": resized,
            "quality_metrics": quality_metrics,
            "metadata": {
                "original_size": f"{original_size[1]}x{original_size[0]}",
                "processed_size": f"{self.TARGET_SIZE[0]}x{self.TARGET_SIZE[1]}",
                "enhancement_applied": apply_enhancement,
                "normalization": "medical" if self.use_medical_norm else "imagenet"
            }
        }
