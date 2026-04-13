"""
DenseNet121 CheXNet Training Engine.

Complete training pipeline for chest X-ray multi-label classification.
Implements:
    - DenseNet121 with ImageNet transfer learning
    - Weighted Binary Cross-Entropy for class imbalance
    - Cosine annealing learning rate schedule
    - Mixed precision training
    - Gradient accumulation
    - Model checkpointing & early stopping
    - TensorBoard logging
    - AUROC metric tracking per pathology

Usage:
    python -m app.training.train_model --data_dir ./data/chexpert --epochs 50 --batch_size 32

Reference:
    CheXNet: Rajpurkar et al., arXiv:1711.05225 (2017)
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def build_densenet121_model(
    num_classes: int = 14,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    freeze_base_layers: Optional[int] = None,
    use_pretrained: bool = True
):
    """
    Build the DenseNet121-based CheXNet model.
    
    Architecture:
        DenseNet121 backbone (ImageNet pretrained)
        → Global Average Pooling
        → Dense(512, ReLU) + Dropout
        → Dense(256, ReLU) + Dropout  
        → Dense(14, Sigmoid) — multi-label output
    
    Args:
        num_classes: Number of pathology classes (default: 14)
        input_shape: Input image dimensions
        dropout_rate: Dropout probability
        freeze_base_layers: If set, freeze first N layers of backbone
        use_pretrained: Use ImageNet pretrained weights
    
    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    
    logger.info("Building DenseNet121 CheXNet architecture...")
    
    # Load DenseNet121 backbone
    weights = 'imagenet' if use_pretrained else None
    base_model = tf.keras.applications.DenseNet121(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        pooling=None  # We'll add our own pooling
    )
    
    # Freeze layers if specified (for fine-tuning)
    if freeze_base_layers is not None:
        for layer in base_model.layers[:freeze_base_layers]:
            layer.trainable = False
        logger.info(f"Frozen first {freeze_base_layers} layers of DenseNet121")
    
    # Build classification head
    inputs = base_model.input
    
    # Feature extraction
    x = base_model.output
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    
    # Batch normalization
    x = tf.keras.layers.BatchNormalization(name='bn_head')(x)
    
    # Dense block 1
    x = tf.keras.layers.Dense(512, name='dense_1')(x)
    x = tf.keras.layers.ReLU(name='relu_1')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout_1')(x)
    
    # Dense block 2
    x = tf.keras.layers.Dense(256, name='dense_2')(x)
    x = tf.keras.layers.ReLU(name='relu_2')(x)
    x = tf.keras.layers.Dropout(dropout_rate * 0.5, name='dropout_2')(x)
    
    # Multi-label output (sigmoid for independent probabilities)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='sigmoid',
        name='predictions',
        kernel_initializer='glorot_uniform',
        bias_initializer=tf.keras.initializers.Constant(-1.0)  # Start with low predictions
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CheXNet_DenseNet121')
    
    total_params = model.count_params()
    trainable_params = sum(
        np.prod(w.shape) for w in model.trainable_weights
    )
    
    logger.info(f"Model built: {total_params:,} total params, {trainable_params:,} trainable")
    
    return model


def weighted_binary_crossentropy(class_weights):
    """
    Weighted Binary Cross-Entropy loss for multi-label classification.
    
    Handles class imbalance by weighting positive samples more heavily
    for rare pathologies.
    
    Args:
        class_weights: Array of shape (num_classes,) with weight per class
    """
    import tensorflow as tf
    
    weights = tf.constant(class_weights, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Binary cross-entropy
        bce = -(
            weights * y_true * tf.math.log(y_pred) +
            (1 - y_true) * tf.math.log(1 - y_pred)
        )
        
        return tf.reduce_mean(bce)
    
    return loss_fn


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling extreme class imbalance.
    
    Focuses training on hard-to-classify examples by down-weighting
    easy negative samples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    import tensorflow as tf
    
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Compute focal weight
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, gamma)
        
        # Binary cross-entropy with focal weight
        bce = -(
            y_true * tf.math.log(y_pred) +
            (1 - y_true) * tf.math.log(1 - y_pred)
        )
        
        loss = alpha * focal_weight * bce
        return tf.reduce_mean(loss)
    
    return loss_fn


class CosineAnnealingSchedule(object):
    """
    Cosine Annealing Learning Rate Schedule with Warm Restarts.
    
    Implements SGDR: Stochastic Gradient Descent with Warm Restarts
    (Loshchilov & Hutter, 2017)
    """
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_epochs: int = 5,
        total_epochs: int = 50
    ):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def __call__(self, epoch):
        import math
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.min_lr + (self.initial_lr - self.min_lr) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class AUROCMetric(object):
    """
    Per-class AUROC (Area Under ROC Curve) metric.
    
    Computes AUROC for each pathology class independently,
    which is the standard evaluation metric for CheXpert/NIH.
    """
    
    def __init__(self, num_classes: int = 14, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_labels = []
    
    def update(self, predictions, labels):
        self.all_predictions.append(predictions)
        self.all_labels.append(labels)
    
    def compute(self) -> Dict:
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        predictions = np.concatenate(self.all_predictions, axis=0)
        labels = np.concatenate(self.all_labels, axis=0)
        
        auroc_scores = {}
        auprc_scores = {}
        
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            
            # AUROC requires at least one positive and one negative sample
            unique_labels = np.unique(labels[:, i])
            if len(unique_labels) >= 2:
                auroc_scores[class_name] = float(roc_auc_score(labels[:, i], predictions[:, i]))
                auprc_scores[class_name] = float(average_precision_score(labels[:, i], predictions[:, i]))
            else:
                auroc_scores[class_name] = 0.0
                auprc_scores[class_name] = 0.0
        
        mean_auroc = np.mean(list(auroc_scores.values()))
        mean_auprc = np.mean(list(auprc_scores.values()))
        
        return {
            "mean_auroc": float(mean_auroc),
            "mean_auprc": float(mean_auprc),
            "per_class_auroc": auroc_scores,
            "per_class_auprc": auprc_scores
        }


class TrainingEngine:
    """
    Complete training engine for DenseNet121 CheXNet model.
    
    Features:
        - Transfer learning from ImageNet
        - Two-phase training (frozen then fine-tuning)
        - Mixed precision training
        - Model checkpointing
        - Early stopping
        - TensorBoard logging
        - Per-class AUROC tracking
        - Training history export
    """
    
    def __init__(
        self,
        model,
        train_ds,
        val_ds,
        test_ds=None,
        class_weights=None,
        output_dir: str = "./trained_models",
        experiment_name: str = None,
        use_mixed_precision: bool = False
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.class_weights = class_weights
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"chexnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directories
        self.experiment_dir = self.output_dir / self.experiment_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Enable mixed precision if requested
        if use_mixed_precision:
            import tensorflow as tf
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("Mixed precision training enabled (FP16)")
            except Exception as e:
                logger.warning(f"Mixed precision not available: {e}")
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "learning_rate": [],
            "epoch_times": []
        }
    
    def compile_model(
        self,
        initial_lr: float = 1e-4,
        loss_type: str = "weighted_bce",
        optimizer: str = "adam"
    ):
        """
        Compile the model with appropriate loss and optimizer.
        
        Args:
            initial_lr: Initial learning rate
            loss_type: 'weighted_bce', 'focal', or 'bce'
            optimizer: 'adam', 'adamw', or 'sgd'
        """
        import tensorflow as tf
        
        # Select loss function
        if loss_type == "weighted_bce" and self.class_weights is not None:
            loss = weighted_binary_crossentropy(self.class_weights)
            logger.info("Using Weighted Binary Cross-Entropy loss")
        elif loss_type == "focal":
            loss = focal_loss(alpha=0.25, gamma=2.0)
            logger.info("Using Focal Loss (α=0.25, γ=2.0)")
        else:
            loss = tf.keras.losses.BinaryCrossentropy()
            logger.info("Using standard Binary Cross-Entropy loss")
        
        # Select optimizer
        if optimizer == "adamw":
            opt = tf.keras.optimizers.AdamW(
                learning_rate=initial_lr,
                weight_decay=1e-5
            )
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(
                learning_rate=initial_lr,
                momentum=0.9,
                nesterov=True
            )
        else:
            opt = tf.keras.optimizers.Adam(
                learning_rate=initial_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=True, num_labels=14)
            ]
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer (lr={initial_lr})")
    
    def train(
        self,
        epochs: int = 50,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-7,
        warmup_epochs: int = 3,
        patience: int = 10,
        loss_type: str = "weighted_bce",
        optimizer: str = "adam",
        two_phase: bool = True,
        phase1_epochs: int = 5,
        phase1_freeze_layers: int = 400
    ) -> Dict:
        """
        Run the full training pipeline.
        
        Two-Phase Training Strategy:
        1. Phase 1: Train only classification head (backbone frozen)
        2. Phase 2: Fine-tune entire network with lower learning rate
        
        Args:
            epochs: Total training epochs
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate for cosine schedule  
            warmup_epochs: Number of warmup epochs
            patience: Early stopping patience
            loss_type: Loss function type
            optimizer: Optimizer type
            two_phase: Enable two-phase training
            phase1_epochs: Epochs for phase 1 (frozen backbone)
            phase1_freeze_layers: Number of backbone layers to freeze in phase 1
        """
        import tensorflow as tf
        
        logger.info("=" * 70)
        logger.info("  AI MedScan — DenseNet121 CheXNet Training")
        logger.info("=" * 70)
        logger.info(f"  Experiment: {self.experiment_name}")
        logger.info(f"  Output: {self.experiment_dir}")
        logger.info(f"  Epochs: {epochs} | LR: {initial_lr} | Patience: {patience}")
        logger.info(f"  Two-Phase: {two_phase} | Loss: {loss_type}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # ── Phase 1: Train classification head only ──
        if two_phase:
            logger.info("\n🔒 PHASE 1: Training classification head (backbone frozen)")
            
            # Freeze backbone
            for layer in self.model.layers:
                if hasattr(layer, 'trainable'):
                    layer.trainable = False
            
            # Unfreeze classification head
            head_layers = ['gap', 'bn_head', 'dense_1', 'relu_1', 'dropout_1',
                          'dense_2', 'relu_2', 'dropout_2', 'predictions']
            for name in head_layers:
                try:
                    self.model.get_layer(name).trainable = True
                except ValueError:
                    pass
            
            self.compile_model(initial_lr=initial_lr * 10, loss_type=loss_type, optimizer=optimizer)
            
            phase1_history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=phase1_epochs,
                callbacks=[
                    tf.keras.callbacks.TensorBoard(log_dir=str(self.log_dir / "phase1")),
                ],
                verbose=1
            )
            
            logger.info(f"Phase 1 complete. Val Loss: {phase1_history.history['val_loss'][-1]:.4f}")
            
            # ── Phase 2: Fine-tune entire network ──
            logger.info("\n🔓 PHASE 2: Fine-tuning entire network")
            
            # Unfreeze all layers
            for layer in self.model.layers:
                if hasattr(layer, 'trainable'):
                    layer.trainable = True
            
            remaining_epochs = epochs - phase1_epochs
        else:
            remaining_epochs = epochs
        
        # Compile for main training
        self.compile_model(initial_lr=initial_lr, loss_type=loss_type, optimizer=optimizer)
        
        # ── Callbacks ──
        callbacks = [
            # Model checkpoint - save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "best_model.keras"),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Also save latest model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "latest_model.keras"),
                save_best_only=False,
                save_weights_only=False,
                verbose=0
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate schedule
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingSchedule(
                    initial_lr=initial_lr,
                    min_lr=min_lr,
                    warmup_epochs=warmup_epochs,
                    total_epochs=remaining_epochs
                ),
                verbose=0
            ),
            
            # Reduce LR on plateau (backup)
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=min_lr,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir / "phase2"),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                str(self.experiment_dir / "training_log.csv"),
                separator=',',
                append=True
            )
        ]
        
        # ── Main Training ──
        logger.info(f"\n🚀 Starting main training for {remaining_epochs} epochs...")
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=remaining_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        total_time = time.time() - start_time
        
        # ── Save final model ──
        final_model_path = str(self.experiment_dir / "final_model.keras")
        self.model.save(final_model_path)
        logger.info(f"\n💾 Final model saved: {final_model_path}")
        
        # Also save as SavedModel format
        savedmodel_path = str(self.experiment_dir / "saved_model")
        try:
            self.model.export(savedmodel_path)
            logger.info(f"💾 SavedModel exported: {savedmodel_path}")
        except Exception as e:
            logger.warning(f"SavedModel export skipped: {e}")
        
        # ── Evaluate on test set ──
        test_results = None
        if self.test_ds is not None:
            logger.info("\n📊 Evaluating on test set...")
            test_results = self._evaluate(self.test_ds)
        
        # ── Save training results ──
        results = {
            "experiment_name": self.experiment_name,
            "total_training_time_seconds": round(total_time, 1),
            "total_training_time_formatted": str(datetime.utcfromtimestamp(total_time).strftime('%H:%M:%S')),
            "total_epochs_trained": len(history.history['loss']),
            "best_val_auc": float(max(history.history.get('val_auc', [0]))),
            "best_val_loss": float(min(history.history['val_loss'])),
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "test_results": test_results,
            "model_path": final_model_path,
            "config": {
                "epochs": epochs,
                "initial_lr": initial_lr,
                "loss_type": loss_type,
                "optimizer": optimizer,
                "two_phase": two_phase
            }
        }
        
        results_path = self.experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"  ✅ Training Complete!")
        logger.info(f"  Time: {results['total_training_time_formatted']}")
        logger.info(f"  Best Val AUC: {results['best_val_auc']:.4f}")
        logger.info(f"  Best Val Loss: {results['best_val_loss']:.4f}")
        if test_results:
            logger.info(f"  Test Mean AUROC: {test_results.get('mean_auroc', 'N/A')}")
        logger.info(f"  Model: {final_model_path}")
        logger.info(f"{'='*70}")
        
        return results
    
    def _evaluate(self, dataset) -> Dict:
        """Evaluate model and compute per-class AUROC."""
        from app.training.dataset import PATHOLOGY_LABELS
        
        auroc_metric = AUROCMetric(
            num_classes=14,
            class_names=PATHOLOGY_LABELS
        )
        
        for batch_x, batch_y in dataset:
            predictions = self.model.predict(batch_x, verbose=0)
            auroc_metric.update(predictions, batch_y.numpy())
        
        results = auroc_metric.compute()
        
        logger.info("\nPer-Class AUROC Scores:")
        logger.info("-" * 40)
        for cls, score in results["per_class_auroc"].items():
            bar = "█" * int(score * 20)
            logger.info(f"  {cls:>20s}: {score:.4f} |{bar}")
        logger.info(f"\n  {'Mean AUROC':>20s}: {results['mean_auroc']:.4f}")
        
        return results


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="AI MedScan — Train DenseNet121 CheXNet Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python -m app.training.train_model --data_dir ./data/chexpert
  
  # Train with custom settings
  python -m app.training.train_model --data_dir ./data/nih --epochs 100 --lr 0.0001 --batch_size 16
  
  # Train with focal loss
  python -m app.training.train_model --data_dir ./data --loss focal --optimizer adamw
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Output directory for models')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--loss', type=str, default='weighted_bce',
                        choices=['weighted_bce', 'focal', 'bce'],
                        help='Loss function')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--no_two_phase', action='store_true',
                        help='Disable two-phase training')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training (FP16)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    import tensorflow as tf
    
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            logger.info(f"  GPU: {gpu.name}")
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Load dataset
    from app.training.dataset import ChestXRayDataset, MedicalDatasetConfig
    
    config = MedicalDatasetConfig(
        dataset_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed
    )
    
    dataset = ChestXRayDataset(config)
    dataset.load_from_directory()
    
    # Print dataset statistics
    stats = dataset.get_dataset_statistics()
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total images: {stats['total_images']}")
    for label, count in stats['positive_counts'].items():
        prevalence = stats['prevalence'][label]
        logger.info(f"  {label:>20s}: {count:>6d} ({prevalence:>5.1f}%)")
    
    # Create TF datasets
    train_ds, val_ds, test_ds, class_weights = dataset.create_tf_datasets()
    
    # Build model
    model = build_densenet121_model(
        num_classes=14,
        dropout_rate=0.3,
        use_pretrained=True
    )
    
    model.summary(print_fn=lambda x: logger.info(x))
    
    # Initialize training engine
    engine = TrainingEngine(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_weights=class_weights,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        use_mixed_precision=args.mixed_precision
    )
    
    # Start training
    results = engine.train(
        epochs=args.epochs,
        initial_lr=args.lr,
        patience=args.patience,
        loss_type=args.loss,
        optimizer=args.optimizer,
        two_phase=not args.no_two_phase
    )
    
    logger.info("\n🎉 Training pipeline complete!")
    logger.info(f"Results saved to: {engine.experiment_dir / 'results.json'}")
    
    return results


if __name__ == "__main__":
    main()
