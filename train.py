"""Main training script for PHME data challenge."""

import argparse
import logging
from pathlib import Path
import mlflow
import joblib

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.training.trainer import XGBoostTrainer


def train_task1(config, data_loader, feature_engineer, logger):
    """Train Task 1: Defect Detection."""
    logger.info("=" * 80)
    logger.info("TASK 1: DEFECT DETECTION")
    logger.info("=" * 80)

    # Load data
    spi_data, aoi_data = data_loader.load_all_data()

    # Prepare data
    df = data_loader.prepare_spi_for_task1(spi_data, aoi_data)

    # Prepare features
    X, y, encoder = feature_engineer.prepare_features_task1(df, fit=True)

    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Initialize trainer
    trainer = XGBoostTrainer(
        config=config,
        task_name="task1",
        pos_label=config.get('task1.pos_label', 0)
    )

    with mlflow.start_run(run_name="task1_complete"):
        # Optimize hyperparameters
        n_trials = config.get('task1.optuna_trials', 50)
        cv_folds = config.get('task1.cv_folds', 5)

        best_params = trainer.optimize_hyperparameters(
            X, y,
            n_trials=n_trials,
            cv_folds=cv_folds
        )

        # Train final model
        model = trainer.train(X, y, params=best_params)

        # Save model and encoder
        model_path = config.get('models.task1.model', 'models/task1_model.json')
        encoder_path = config.get('models.task1.encoder', 'models/task1_encoder.pkl')

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        feature_engineer.save_encoder("task1_encoder", encoder_path)

        mlflow.log_artifacts(Path(model_path).parent)

        logger.info(f"Task 1 model saved to {model_path}")
        logger.info(f"Task 1 encoder saved to {encoder_path}")


def train_task2(config, data_loader, feature_engineer, logger):
    """Train Task 2: Operator Label Classification (Inner and Outer models)."""
    logger.info("=" * 80)
    logger.info("TASK 2: OPERATOR LABEL CLASSIFICATION")
    logger.info("=" * 80)

    # Load data
    spi_data, aoi_data = data_loader.load_all_data()

    # Train Inner Model
    logger.info("-" * 80)
    logger.info("Training Inner Model (with PinNumber)")
    logger.info("-" * 80)

    df_inner = data_loader.prepare_data_for_task2(spi_data, aoi_data, inner=True)
    X_inner, y_inner, encoder_inner = feature_engineer.prepare_features_task2(
        df_inner, inner=True, fit=True
    )

    logger.info(f"Inner model data shape: X={X_inner.shape}, y={y_inner.shape}")

    trainer_inner = XGBoostTrainer(
        config=config,
        task_name="task2_inner",
        pos_label=config.get('task2.pos_label', 0)
    )

    with mlflow.start_run(run_name="task2_inner_complete"):
        best_params_inner = trainer_inner.optimize_hyperparameters(
            X_inner, y_inner,
            n_trials=config.get('task2.optuna_trials', 50),
            cv_folds=config.get('task2.cv_folds', 5)
        )

        model_inner = trainer_inner.train(X_inner, y_inner, params=best_params_inner)

        model_path = config.get('models.task2_inner.model', 'models/task2_inner_model.json')
        encoder_path = config.get('models.task2_inner.encoder', 'models/task2_inner_encoder.pkl')

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer_inner.save_model(model_path)
        feature_engineer.save_encoder("task2_inner_encoder", encoder_path)

        logger.info(f"Task 2 inner model saved to {model_path}")

    # Train Outer Model
    logger.info("-" * 80)
    logger.info("Training Outer Model (without PinNumber)")
    logger.info("-" * 80)

    df_outer = data_loader.prepare_data_for_task2(spi_data, aoi_data, inner=False)
    X_outer, y_outer, encoder_outer = feature_engineer.prepare_features_task2(
        df_outer, inner=False, fit=True
    )

    logger.info(f"Outer model data shape: X={X_outer.shape}, y={y_outer.shape}")

    trainer_outer = XGBoostTrainer(
        config=config,
        task_name="task2_outer",
        pos_label=config.get('task2.pos_label', 0)
    )

    with mlflow.start_run(run_name="task2_outer_complete"):
        best_params_outer = trainer_outer.optimize_hyperparameters(
            X_outer, y_outer,
            n_trials=config.get('task2.optuna_trials', 50),
            cv_folds=config.get('task2.cv_folds', 5)
        )

        model_outer = trainer_outer.train(X_outer, y_outer, params=best_params_outer)

        model_path = config.get('models.task2_outer.model', 'models/task2_outer_model.json')
        encoder_path = config.get('models.task2_outer.encoder', 'models/task2_outer_encoder.pkl')

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer_outer.save_model(model_path)
        feature_engineer.save_encoder("task2_outer_encoder", encoder_path)

        logger.info(f"Task 2 outer model saved to {model_path}")


def train_task3(config, data_loader, feature_engineer, logger):
    """Train Task 3: Repair Label Classification."""
    logger.info("=" * 80)
    logger.info("TASK 3: REPAIR LABEL CLASSIFICATION")
    logger.info("=" * 80)

    # Load data
    spi_data, aoi_data = data_loader.load_all_data()

    # Prepare data
    df = data_loader.prepare_data_for_task3(spi_data, aoi_data)

    # Prepare features
    X, y, encoder, aoi_labels = feature_engineer.prepare_features_task3(df, fit=True)

    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Initialize trainer
    trainer = XGBoostTrainer(
        config=config,
        task_name="task3",
        pos_label=1  # NotPossibleToRepair
    )

    with mlflow.start_run(run_name="task3_complete"):
        # Optimize hyperparameters
        best_params = trainer.optimize_hyperparameters(
            X, y,
            n_trials=config.get('task3.optuna_trials', 50),
            cv_folds=config.get('task3.cv_folds', 5)
        )

        # Train final model
        model = trainer.train(X, y, params=best_params)

        # Save model, encoder, and AOI labels
        model_path = config.get('models.task3.model', 'models/task3_model.json')
        encoder_path = config.get('models.task3.encoder', 'models/task3_encoder.pkl')
        aoi_labels_path = config.get('models.task3.aoi_labels', 'models/task3_aoi_labels.pkl')

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_path)
        feature_engineer.save_encoder("task3_encoder", encoder_path)
        joblib.dump(aoi_labels, aoi_labels_path)

        mlflow.log_artifacts(Path(model_path).parent)

        logger.info(f"Task 3 model saved to {model_path}")
        logger.info(f"Task 3 encoder saved to {encoder_path}")
        logger.info(f"Task 3 AOI labels saved to {aoi_labels_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PHME defect detection models')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['task1', 'task2', 'task3', 'all'],
        default='all',
        help='Which task to train'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file'
    )
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(
        name='phme_training',
        log_file=args.log_file,
        level=logging.INFO
    )

    logger.info("=" * 80)
    logger.info("PHME DATA CHALLENGE - MODEL TRAINING")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ConfigLoader(args.config)

    # Initialize components
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)

    # Import numpy here (needed for training functions)
    import numpy as np
    globals()['np'] = np

    # Train selected task(s)
    if args.task in ['task1', 'all']:
        train_task1(config, data_loader, feature_engineer, logger)

    if args.task in ['task2', 'all']:
        train_task2(config, data_loader, feature_engineer, logger)

    if args.task in ['task3', 'all']:
        train_task3(config, data_loader, feature_engineer, logger)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
