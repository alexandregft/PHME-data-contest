"""Training pipeline with MLflow integration."""

import xgboost as xgb
import mlflow
import mlflow.xgboost
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, make_scorer
)
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import joblib

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """XGBoost model trainer with MLflow integration."""

    def __init__(self, config, task_name: str, pos_label: int = 0):
        """
        Initialize trainer.

        Args:
            config: Configuration object
            task_name: Name of the task (e.g., 'task1', 'task2_inner')
            pos_label: Positive label for F1 score calculation
        """
        self.config = config
        self.task_name = task_name
        self.pos_label = pos_label
        self.best_params = None
        self.model = None

        # Setup MLflow
        mlflow.set_tracking_uri(config.get('mlflow.tracking_uri', 'mlruns'))
        mlflow.set_experiment(config.get('mlflow.experiment_name', 'PHME_Defect_Detection'))

    def f1_eval(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[str, float]:
        """
        Custom F1 evaluation metric for XGBoost.

        Args:
            y_pred: Predicted labels
            y_true: True labels

        Returns:
            Tuple of (metric name, metric value)
        """
        score = f1_score(y_true, y_pred, pos_label=self.pos_label)
        return 'f1_score', score

    def objective(
        self,
        trial: Trial,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds

        Returns:
            Mean F1 score across folds
        """
        # Suggest hyperparameters
        params = {
            'objective': trial.suggest_categorical('objective', ['binary:logistic']),
            'tree_method': trial.suggest_categorical('tree_method', ['hist']),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_categorical(
                'n_estimators',
                [30, 40, 50, 70, 100, 150, 200]
            ),
            'max_depth': trial.suggest_categorical(
                'max_depth',
                [3, 5, 7, 9, 11, 13, 15, 17, 20]
            ),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            'random_state': self.config.get('training.random_state', 42),
            'n_jobs': -1,
        }

        # Create model
        model = xgb.XGBClassifier(**params)

        # Cross-validation
        scorer = make_scorer(f1_score, pos_label=self.pos_label)
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scorer, n_jobs=-1)
        mean_score = scores.mean()

        # Log to MLflow
        mlflow.log_metrics({
            f"cv_f1_mean": mean_score,
            f"cv_f1_std": scores.std(),
        })

        return mean_score

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds

        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization for {self.task_name}...")
        logger.info(f"Running {n_trials} trials with {cv_folds}-fold cross-validation")

        with mlflow.start_run(run_name=f"{self.task_name}_optimization", nested=True):
            # Log task info
            mlflow.log_params({
                "task": self.task_name,
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "pos_label": self.pos_label,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
            })

            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.get('training.random_state', 42))
            )

            # Optimize
            study.optimize(
                lambda trial: self.objective(trial, X, y, cv_folds),
                n_trials=n_trials,
                show_progress_bar=True
            )

            # Get best parameters
            self.best_params = study.best_trial.params

            # Log best parameters
            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
            mlflow.log_metric("best_cv_f1", study.best_value)

            # Save study
            study_path = f"models/{self.task_name}_study.pkl"
            Path(study_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(study, study_path)
            mlflow.log_artifact(study_path)

            logger.info(f"Best F1 score: {study.best_value:.4f}")
            logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        params: Dict[str, Any] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            params: Model parameters (uses best_params if None)

        Returns:
            Trained XGBoost model
        """
        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided and no optimization performed")
            params = self.best_params

        logger.info(f"Training model for {self.task_name}...")

        with mlflow.start_run(run_name=f"{self.task_name}_training", nested=True):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_params({
                "task": self.task_name,
                "n_train_samples": X_train.shape[0],
                "n_features": X_train.shape[1],
            })

            # Train model
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train)

            # Evaluate on training set
            y_train_pred = self.model.predict(X_train)
            self._log_metrics(y_train, y_train_pred, prefix="train")

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                self._log_metrics(y_val, y_val_pred, prefix="val")
                mlflow.log_param("n_val_samples", X_val.shape[0])

            # Log model
            mlflow.xgboost.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=f"phme_{self.task_name}"
            )

            logger.info(f"Model training completed for {self.task_name}")

        return self.model

    def _log_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""):
        """
        Log classification metrics to MLflow.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            prefix: Metric name prefix
        """
        prefix = f"{prefix}_" if prefix else ""

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Log metrics
        mlflow.log_metrics({
            f"{prefix}accuracy": accuracy,
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1_score": f1,
        })

        # Log confusion matrix elements
        if cm.shape == (2, 2):
            mlflow.log_metrics({
                f"{prefix}tn": int(cm[0, 0]),
                f"{prefix}fp": int(cm[0, 1]),
                f"{prefix}fn": int(cm[1, 0]),
                f"{prefix}tp": int(cm[1, 1]),
            })

        logger.info(f"{prefix}Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")

    def save_model(self, model_path: str):
        """
        Save model to file.

        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON for XGBoost compatibility
        if model_path.endswith('.json'):
            self.model.save_model(model_path)
        else:
            # Save with joblib
            joblib.dump(self.model, model_path)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> xgb.XGBClassifier:
        """
        Load model from file.

        Args:
            model_path: Path to model file

        Returns:
            Loaded XGBoost model
        """
        if model_path.endswith('.json'):
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

        logger.info(f"Model loaded from {model_path}")
        return self.model

    def cross_validate_by_panel(
        self,
        df: pd.DataFrame,
        feature_engineer,
        n_folds: int = 5,
        task_type: str = "task1"
    ) -> Dict[str, float]:
        """
        Perform cross-validation by panel ID.

        Args:
            df: Input DataFrame with PanelID
            feature_engineer: FeatureEngineer instance
            n_folds: Number of folds
            task_type: Type of task ('task1', 'task2_inner', 'task2_outer', 'task3')

        Returns:
            Dictionary with mean and std of F1 scores
        """
        import pandas as pd
        import random

        logger.info(f"Performing {n_folds}-fold cross-validation by panel for {self.task_name}")

        # Get unique panel IDs
        panel_ids = df['PanelID'].unique().tolist()
        random.shuffle(panel_ids)

        # Create folds
        fold_size = len(panel_ids) // n_folds
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(panel_ids)
            folds.append(panel_ids[start_idx:end_idx])

        f1_scores = []

        with mlflow.start_run(run_name=f"{self.task_name}_cv_by_panel", nested=True):
            mlflow.log_params({
                "task": self.task_name,
                "n_folds": n_folds,
                "n_panels": len(panel_ids),
            })

            for fold_idx in range(n_folds):
                logger.info(f"Processing fold {fold_idx + 1}/{n_folds}")

                # Split data
                test_panels = folds[fold_idx]
                train_panels = [p for i, fold in enumerate(folds) if i != fold_idx for p in fold]

                df_train = df[df['PanelID'].isin(train_panels)].copy()
                df_test = df[df['PanelID'].isin(test_panels)].copy()

                # Prepare features based on task type
                if task_type == "task1":
                    X_train, y_train, encoder = feature_engineer.prepare_features_task1(
                        df_train, fit=True
                    )
                    X_test, y_test, _ = feature_engineer.prepare_features_task1(
                        df_test, encoder=encoder, fit=False
                    )
                elif task_type == "task2_inner":
                    X_train, y_train, encoder = feature_engineer.prepare_features_task2(
                        df_train, inner=True, fit=True
                    )
                    X_test, y_test, _ = feature_engineer.prepare_features_task2(
                        df_test, inner=True, encoder=encoder, fit=False
                    )
                elif task_type == "task2_outer":
                    X_train, y_train, encoder = feature_engineer.prepare_features_task2(
                        df_train, inner=False, fit=True
                    )
                    X_test, y_test, _ = feature_engineer.prepare_features_task2(
                        df_test, inner=False, encoder=encoder, fit=False
                    )
                else:  # task3
                    X_train, y_train, encoder, aoi_labels = feature_engineer.prepare_features_task3(
                        df_train, fit=True
                    )
                    X_test, y_test, _, _ = feature_engineer.prepare_features_task3(
                        df_test, encoder=encoder, fit=False, aoi_labels=aoi_labels
                    )

                # Train model
                params = self.best_params if self.best_params else self._get_default_params()
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate F1 score
                f1 = f1_score(y_test, y_pred, pos_label=self.pos_label)
                f1_scores.append(f1)

                mlflow.log_metric(f"fold_{fold_idx}_f1", f1)
                logger.info(f"Fold {fold_idx + 1} F1 score: {f1:.4f}")

            # Log overall metrics
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)

            mlflow.log_metrics({
                "cv_panel_f1_mean": mean_f1,
                "cv_panel_f1_std": std_f1,
            })

            logger.info(f"Cross-validation completed: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        return {"mean": mean_f1, "std": std_f1, "scores": f1_scores}

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        return {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'learning_rate': 0.01,
            'n_estimators': 100,
            'max_depth': 7,
            'min_child_weight': 1,
            'lambda': 1.0,
            'alpha': 0.0,
            'colsample_bytree': 1.0,
            'random_state': self.config.get('training.random_state', 42),
            'n_jobs': -1,
        }
