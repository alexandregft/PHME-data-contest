"""Inference module for PHME classification tasks."""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import List, Tuple
import logging

from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PHMEPredictor:
    """Predictor for all three PHME classification tasks."""

    def __init__(self, config):
        """
        Initialize predictor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)

        # Model components
        self.task1_model = None
        self.task1_encoder = None

        self.task2_inner_model = None
        self.task2_inner_encoder = None
        self.task2_outer_model = None
        self.task2_outer_encoder = None

        self.task3_model = None
        self.task3_encoder = None
        self.task3_aoi_labels = None

    def load_models(self, models_dir: str = "models"):
        """
        Load all trained models and encoders.

        Args:
            models_dir: Directory containing model files
        """
        models_path = Path(models_dir)

        logger.info("Loading models and encoders...")

        # Task 1
        task1_encoder_path = models_path / "task1_encoder.pkl"
        task1_model_path = models_path / "task1_model.json"

        if task1_encoder_path.exists() and task1_model_path.exists():
            self.task1_encoder = self.feature_engineer.load_encoder(str(task1_encoder_path))
            self.task1_model = xgb.XGBClassifier()
            self.task1_model.load_model(str(task1_model_path))
            logger.info("Task 1 model loaded")

        # Task 2 Inner
        task2_inner_encoder_path = models_path / "task2_inner_encoder.pkl"
        task2_inner_model_path = models_path / "task2_inner_model.json"

        if task2_inner_encoder_path.exists() and task2_inner_model_path.exists():
            self.task2_inner_encoder = self.feature_engineer.load_encoder(
                str(task2_inner_encoder_path)
            )
            self.task2_inner_model = xgb.XGBClassifier()
            self.task2_inner_model.load_model(str(task2_inner_model_path))
            logger.info("Task 2 inner model loaded")

        # Task 2 Outer
        task2_outer_encoder_path = models_path / "task2_outer_encoder.pkl"
        task2_outer_model_path = models_path / "task2_outer_model.json"

        if task2_outer_encoder_path.exists() and task2_outer_model_path.exists():
            self.task2_outer_encoder = self.feature_engineer.load_encoder(
                str(task2_outer_encoder_path)
            )
            self.task2_outer_model = xgb.XGBClassifier()
            self.task2_outer_model.load_model(str(task2_outer_model_path))
            logger.info("Task 2 outer model loaded")

        # Task 3
        task3_encoder_path = models_path / "task3_encoder.pkl"
        task3_model_path = models_path / "task3_model.json"
        task3_aoi_labels_path = models_path / "task3_aoi_labels.pkl"

        if all([task3_encoder_path.exists(), task3_model_path.exists(),
                task3_aoi_labels_path.exists()]):
            self.task3_encoder = self.feature_engineer.load_encoder(str(task3_encoder_path))
            self.task3_model = xgb.XGBClassifier()
            self.task3_model.load_model(str(task3_model_path))
            self.task3_aoi_labels = joblib.load(task3_aoi_labels_path)
            logger.info("Task 3 model loaded")

    def classification_1(self, spi: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """
        Task 1: Detect defects from SPI data.

        Args:
            spi: SPI DataFrame

        Returns:
            List of tuples (PanelID, FigureID, ComponentID) for detected defects
        """
        if self.task1_model is None or self.task1_encoder is None:
            raise ValueError("Task 1 model not loaded")

        logger.info("Running Task 1 prediction...")

        # Prepare data
        spi_real = spi.copy()
        spi_real = spi_real.dropna()
        spi_real["ComponentID2"] = spi_real["ComponentID"]
        spi_real["FigureID2"] = spi_real["FigureID"]

        # Get numeric features
        num_features_to_cat = self.config.get('features.numeric_to_categorical')
        num_features = self.config.get('features.numeric_features')

        # Convert numeric features
        for feat in num_features:
            spi_real[feat] = spi_real[feat].astype('float')

        # Encode ComponentID
        spi_real["ComponentID2"] = spi_real["ComponentID"].astype('category').cat.codes

        # Apply encoder
        list_var_to_encode = ["ComponentID2"]
        AE_val = self.task1_encoder.transform(spi_real[list_var_to_encode].copy())
        AE_val.columns = [x + "_encoded" for x in AE_val.columns]
        spi_real = pd.concat([spi_real, AE_val], axis=1)

        # Select features
        features_x = [f"{x}_encoded" for x in list_var_to_encode] + \
                    num_features_to_cat + num_features
        X_test = spi_real[features_x].fillna(0).to_numpy()

        # Predict
        pred = self.task1_model.predict(X_test)
        spi_real['Pred_label'] = pred

        # Filter defects (label 0)
        spi_real = spi_real.loc[spi_real['Pred_label'] == 0]
        spi_real = spi_real.dropna(subset=['FigureID2'])

        spi_real["FigureID2"] = spi_real["FigureID2"].astype(int).astype(str)

        # Extract defect list
        defects = list(spi_real[["PanelID", "FigureID2", "ComponentID"]].itertuples(
            index=False, name=None
        ))

        logger.info(f"Task 1: Detected {len(defects)} defects")

        return defects

    def classification_2(
        self,
        spi: pd.DataFrame,
        aoi: pd.DataFrame
    ) -> List[Tuple[str, str, str, str]]:
        """
        Task 2: Classify OperatorLabel (Good/Bad).

        Args:
            spi: SPI DataFrame
            aoi: AOI DataFrame (without OperatorLabel and RepairLabel)

        Returns:
            List of tuples (PanelID, FigureID, ComponentID, PredictedOperatorLabel)
        """
        if any([self.task2_inner_model is None, self.task2_inner_encoder is None,
                self.task2_outer_model is None, self.task2_outer_encoder is None]):
            raise ValueError("Task 2 models not loaded")

        logger.info("Running Task 2 prediction...")

        spi_real = spi.copy()
        aoi_real = aoi.copy()

        # Prepare AOI data
        spi_real['PinNumber'] = spi_real['PinNumber'].astype(str)
        aoi_real['PinNumber'] = aoi_real['PinNumber'].astype('Int64').astype(str)
        aoi_real['FigureID2'] = aoi_real['FigureID']
        aoi_real['ComponentID2'] = aoi_real['ComponentID']

        # Add count features
        aoi_real['Count_Pin'] = aoi_real.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )["PinNumber"].transform('count')
        aoi_real['Count_Pin_Figure'] = aoi_real.groupby(
            ["PanelID", "FigureID2"]
        )["PinNumber"].transform('count')

        spi_real = spi_real.dropna()

        # Process inner model (with PinNumber)
        aoi_inner = aoi_real[aoi_real['PinNumber'] != '<NA>'].copy()
        aoi_inner = self._predict_task2_inner(spi_real, aoi_inner)

        # Process outer model (without PinNumber)
        aoi_outer = aoi_real[aoi_real['PinNumber'] == '<NA>'].copy()
        aoi_outer = self._predict_task2_outer(spi_real, aoi_outer)

        # Combine predictions
        aoi_combined = pd.concat([aoi_outer, aoi_inner])

        # Aggregate at component level (take minimum prediction - most conservative)
        aoi_group = aoi_combined.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )['Pred'].min().reset_index()

        # Map predictions to labels
        aoi_group['Pred'] = aoi_group['Pred'].map({0: "Bad", 1: "Good"})

        # Extract prediction list
        predicted = list(aoi_group[[
            "PanelID", "FigureID2", "ComponentID2", 'Pred'
        ]].itertuples(index=False, name=None))

        logger.info(f"Task 2: Predicted {len(predicted)} components")

        return predicted

    def _predict_task2_inner(
        self,
        spi: pd.DataFrame,
        aoi_inner: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict using inner model for Task 2."""
        # Merge with SPI
        aoi_inner = aoi_inner.merge(
            spi,
            on=['PanelID', 'FigureID', "ComponentID", "PinNumber"],
            how="inner"
        )

        # Add composite features
        aoi_inner['FigureID_ComponentID'] = (
            aoi_inner['FigureID'].astype(str) + '_' +
            aoi_inner['ComponentID'].astype(str)
        ).astype("category").cat.codes

        # Encode variables
        list_var_to_encode = self.config.get('task2.inner.encoding_variables')
        for var in list_var_to_encode:
            aoi_inner[var] = aoi_inner[var].astype('category').cat.codes

        # Apply encoder
        AE_val = self.task2_inner_encoder.transform(aoi_inner[list_var_to_encode].copy())
        AE_val.columns = [x + "_encoded" for x in AE_val.columns]
        aoi_inner = pd.concat([aoi_inner, AE_val], axis=1)

        # Select features
        num_features = self.config.get('features.numeric_features')
        num_features_to_cat = self.config.get('features.numeric_to_categorical')
        col_features = [x + "_encoded" for x in list_var_to_encode] + \
                      num_features + num_features_to_cat + \
                      ["Count_Pin", "Count_Pin_Figure"]

        X_test = aoi_inner[col_features].fillna(0).to_numpy()

        # Predict
        aoi_inner['Pred'] = self.task2_inner_model.predict(X_test)

        return aoi_inner

    def _predict_task2_outer(
        self,
        spi: pd.DataFrame,
        aoi_outer: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict using outer model for Task 2."""
        # Aggregate SPI features
        features_to_bin = self.config.get('features.numeric_to_categorical')
        InputSPI_grouped = spi.groupby(
            ["PanelID", "FigureID", "ComponentID"]
        )[features_to_bin].mean().reset_index()

        # Merge with AOI
        aoi_outer = aoi_outer.merge(
            InputSPI_grouped,
            on=['PanelID', 'FigureID', "ComponentID"],
            how="left"
        )

        # Add composite features
        aoi_outer['FigureID_ComponentID'] = (
            aoi_outer['FigureID'].astype(str) + '_' +
            aoi_outer['ComponentID'].astype(str)
        ).astype("category").cat.codes

        # Encode variables
        list_var_to_encode = self.config.get('task2.outer.encoding_variables')
        for var in list_var_to_encode:
            aoi_outer[var] = aoi_outer[var].astype('category').cat.codes

        # Apply encoder
        AE_val = self.task2_outer_encoder.transform(aoi_outer[list_var_to_encode].copy())
        AE_val.columns = [x + "_encoded" for x in AE_val.columns]
        aoi_outer = pd.concat([aoi_outer, AE_val], axis=1)

        # Select features
        col_features = features_to_bin + \
                      [x + "_encoded" for x in list_var_to_encode] + \
                      ["Count_Pin", "Count_Pin_Figure"]

        X_test = aoi_outer[col_features].fillna(0).to_numpy()

        # Predict
        aoi_outer['Pred'] = self.task2_outer_model.predict(X_test)

        return aoi_outer

    def classification_3(
        self,
        spi: pd.DataFrame,
        aoi: pd.DataFrame
    ) -> List[Tuple[str, str, str, str]]:
        """
        Task 3: Classify RepairLabel (FalseScrap/NotPossibleToRepair).

        Args:
            spi: SPI DataFrame
            aoi: AOI DataFrame (with OperatorLabel, without RepairLabel)

        Returns:
            List of tuples (PanelID, FigureID, ComponentID, PredictedRepairLabel)
        """
        if any([self.task3_model is None, self.task3_encoder is None,
                self.task3_aoi_labels is None]):
            raise ValueError("Task 3 model not loaded")

        logger.info("Running Task 3 prediction...")

        spi_real = spi.copy()
        aoi_real = aoi.copy()

        # Filter only bad components
        aoi_real = aoi_real.loc[aoi_real.OperatorLabel == "Bad"]

        # Prepare AOI data
        aoi_real['PinNumber'] = aoi_real['PinNumber'].astype('Int64').astype(str)
        aoi_real['FigureID2'] = aoi_real['FigureID']
        aoi_real['ComponentID2'] = aoi_real['ComponentID']

        # Add count features
        aoi_real['Count_Pin'] = aoi_real.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )["PinNumber"].transform('count')
        aoi_real['Count_Pin_Figure'] = aoi_real.groupby(
            ["PanelID", "FigureID2"]
        )["PinNumber"].transform('count')
        aoi_real['Count_Pin_Panel'] = aoi_real.groupby(
            ["PanelID"]
        )["PinNumber"].transform('count')

        # Machine ID encoding
        aoi_real["MachineID"] = aoi_real["MachineID"].astype('category').cat.codes

        # Create keys
        spi_real['key_spi'] = (
            spi_real['PanelID'].astype(str) + '_' +
            spi_real['FigureID'].astype(int, errors='ignore').astype(str) + '_' +
            spi_real['ComponentID'].astype(str)
        )
        aoi_real['key_spi'] = (
            aoi_real['PanelID'].astype(str) + '_' +
            aoi_real['FigureID'].astype(int, errors='ignore').astype(str) + '_' +
            aoi_real['ComponentID'].astype(str)
        )

        # Create AOI label features
        aoi_real['PanelID_FigureID_ComponentID'] = aoi_real['key_spi']

        for element in self.task3_aoi_labels:
            aoi_real[element] = 0

        for item in aoi_real['PanelID_FigureID_ComponentID'].unique():
            df_inter = aoi_real.loc[aoi_real['PanelID_FigureID_ComponentID'] == item]
            list_item_aoilabel = df_inter.AOILabel.unique()

            for element in self.task3_aoi_labels:
                if element in list_item_aoilabel:
                    aoi_real.loc[aoi_real['PanelID_FigureID_ComponentID'] == item, element] = 1

        # Prepare SPI data
        spi_real = spi_real.dropna(subset=['FigureID'])
        spi_real['FigureID2'] = spi_real['FigureID']
        spi_real['ComponentID2'] = spi_real['ComponentID']
        spi_real = spi_real.loc[spi_real['key_spi'].isin(aoi_real['key_spi'].unique())]

        # Aggregate SPI features
        num_features = self.config.get('features.numeric_features')
        features_to_bin = self.config.get('features.numeric_to_categorical')
        spi_real['Shape(um)'] = pd.to_numeric(spi_real['Shape(um)'], errors='coerce')

        InputSPI_grouped = spi_real.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )[features_to_bin + num_features].mean().reset_index()

        # Merge with AOI
        aoi_real = aoi_real.merge(
            InputSPI_grouped,
            how="left",
            left_on=["PanelID", "FigureID2", "ComponentID2"],
            right_on=["PanelID", "FigureID2", "ComponentID2"]
        )
        aoi_real = aoi_real.drop_duplicates(subset=["PanelID", "FigureID2", "ComponentID2"])

        # Add composite features
        aoi_real['FigureID_ComponentID'] = (
            aoi_real['FigureID'].astype(str) + '_' +
            aoi_real['ComponentID'].astype(str)
        ).astype("category").cat.codes

        # Encode variables
        list_var_to_encode = self.config.get('task3.encoding_variables')
        for var in list_var_to_encode:
            aoi_real[var] = aoi_real[var].astype('category').cat.codes

        # Apply encoder
        AE_val = self.task3_encoder.transform(aoi_real[list_var_to_encode].copy())
        AE_val.columns = [x + "_encoded" for x in AE_val.columns]
        X_test_final = pd.concat([aoi_real, AE_val], axis=1)

        # Select features
        col_features = (
            num_features + [x + "_encoded" for x in list_var_to_encode] +
            ["Count_Pin", "Count_Pin_Figure", "Count_Pin_Panel"] +
            self.task3_aoi_labels + features_to_bin + ["MachineID"]
        )

        X_test_final = X_test_final[col_features].fillna(0).to_numpy()

        # Predict
        aoi_real['Pred'] = self.task3_model.predict(X_test_final)

        # Aggregate at component level
        aoi_group = aoi_real.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )['Pred'].min().reset_index()

        # Map predictions to labels
        aoi_group['Pred'] = aoi_group['Pred'].map({
            0: "FalseScrap",
            1: "NotPossibleToRepair"
        })

        # Extract prediction list
        predicted = list(aoi_group[[
            "PanelID", "FigureID2", "ComponentID2", 'Pred'
        ]].itertuples(index=False, name=None))

        logger.info(f"Task 3: Predicted {len(predicted)} components")

        return predicted
