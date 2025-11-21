"""Feature engineering for PHME data challenge."""

import pandas as pd
import numpy as np
import category_encoders as ce
import joblib
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for defect detection tasks."""

    def __init__(self, config):
        """
        Initialize feature engineer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.encoders = {}

    def add_composite_features(self, df: pd.DataFrame, task: str = "task1") -> pd.DataFrame:
        """
        Add composite features for the given task.

        Args:
            df: Input DataFrame
            task: Task identifier ('task1', 'task2', or 'task3')

        Returns:
            DataFrame with additional composite features
        """
        df = df.copy()

        # Component ID encoding
        df["ComponentID2"] = df["ComponentID"].astype('category').cat.codes

        # Composite ID features for task 2 and 3
        if task in ["task2", "task3"]:
            df['FigureID_ComponentID'] = (
                df['FigureID'].astype(str) + '_' + df['ComponentID'].astype(str)
            ).astype("category").cat.codes

        if task == "task2":
            df['FigureID_ComponentID_PinNumber'] = (
                df['FigureID'].astype(str) + '_' +
                df['ComponentID'].astype(str) + '_' +
                df['PinNumber'].astype(str)
            ).astype("category").cat.codes

            df['FigureID_ComponentID_PinNumber_AOILabel'] = (
                df['FigureID'].astype(str) + '_' +
                df['ComponentID'].astype(str) + '_' +
                df['PinNumber'].astype(str) +
                df['AOILabel'].astype(str)
            ).astype("category").cat.codes

            df['FigureID_ComponentID_AOILabel'] = (
                df['FigureID'].astype(str) + '_' +
                df['ComponentID'].astype(str) + '_' +
                df['AOILabel'].astype(str)
            ).astype("category").cat.codes

        return df

    def create_aoi_label_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create one-hot encoded features for AOI labels (Task 3).

        Args:
            df: Input DataFrame with AOILabel column

        Returns:
            Tuple of (DataFrame with AOI label features, list of AOI label names)
        """
        df = df.copy()

        # Create panel-figure-component key
        df['PanelID_FigureID_ComponentID'] = (
            df['PanelID'].astype(str) + '_' +
            df['FigureID'].astype(str) + '_' +
            df['ComponentID'].astype(str)
        )

        # Get unique AOI labels
        aoi_labels = df['AOILabel'].unique().tolist()

        # Initialize label columns
        for label in aoi_labels:
            df[label] = 0

        # Set label presence
        for key in df['PanelID_FigureID_ComponentID'].unique():
            df_sub = df.loc[df['PanelID_FigureID_ComponentID'] == key]
            present_labels = df_sub['AOILabel'].unique()

            for label in aoi_labels:
                if label in present_labels:
                    df.loc[df['PanelID_FigureID_ComponentID'] == key, label] = 1

        logger.info(f"Created {len(aoi_labels)} AOI label features")

        return df, aoi_labels

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        variables: List[str],
        target: pd.Series = None,
        encoder: ce.CatBoostEncoder = None,
        fit: bool = True,
        encoder_name: str = None
    ) -> Tuple[pd.DataFrame, ce.CatBoostEncoder]:
        """
        Encode categorical variables using CatBoost encoder.

        Args:
            df: Input DataFrame
            variables: List of variable names to encode
            target: Target variable (required if fit=True)
            encoder: Pre-fitted encoder (if fit=False)
            fit: Whether to fit the encoder
            encoder_name: Name for storing/retrieving encoder

        Returns:
            Tuple of (DataFrame with encoded features, fitted encoder)
        """
        df = df.copy()

        # Ensure categorical variables are encoded as categories
        for var in variables:
            if var in df.columns:
                df[var] = df[var].astype('category').cat.codes

        if fit:
            if target is None:
                raise ValueError("Target is required when fit=True")

            # Initialize and fit encoder
            a_param = self.config.get('training.catboost_encoder_a', 0.3)
            encoder = ce.CatBoostEncoder(a=a_param)

            encoded = encoder.fit_transform(df[variables].copy(), target)
            encoded.columns = [f"{x}_encoded" for x in encoded.columns]

            # Store encoder
            if encoder_name:
                self.encoders[encoder_name] = encoder

            logger.info(f"Fitted CatBoost encoder for {len(variables)} variables")
        else:
            if encoder is None:
                raise ValueError("Encoder is required when fit=False")

            # Transform using pre-fitted encoder
            encoded = encoder.transform(df[variables].copy())
            encoded.columns = [f"{x}_encoded" for x in encoded.columns]

            logger.info(f"Transformed {len(variables)} variables using pre-fitted encoder")

        # Concatenate encoded features
        df_encoded = pd.concat([df, encoded], axis=1)

        return df_encoded, encoder

    def prepare_features_task1(
        self,
        df: pd.DataFrame,
        encoder: ce.CatBoostEncoder = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, pd.Series, ce.CatBoostEncoder]:
        """
        Prepare features for Task 1.

        Args:
            df: Input DataFrame
            encoder: Pre-fitted encoder (if fit=False)
            fit: Whether to fit the encoder

        Returns:
            Tuple of (feature array, target series, encoder)
        """
        df = df.copy()

        # Get numeric features
        num_features_to_cat = self.config.get('features.numeric_to_categorical')
        num_features = self.config.get('features.numeric_features')

        # Drop rows with missing numeric features
        df = df.dropna(subset=num_features)

        # Convert numeric features to float
        for feat in num_features:
            df[feat] = df[feat].astype('float')

        # Add composite features
        df = self.add_composite_features(df, task="task1")

        # Get target
        target = df['Target'].astype(int)

        # Encode categorical features
        variables_to_encode = ["ComponentID2"]
        df_encoded, encoder = self.encode_categorical_features(
            df,
            variables_to_encode,
            target=target if fit else None,
            encoder=encoder,
            fit=fit,
            encoder_name="task1_encoder" if fit else None
        )

        # Select final features
        feature_cols = [f"{x}_encoded" for x in variables_to_encode] + \
                      num_features_to_cat + num_features
        X = df_encoded[feature_cols].fillna(0).to_numpy()

        logger.info(f"Task 1 features prepared: {X.shape}")

        return X, target, encoder

    def prepare_features_task2(
        self,
        df: pd.DataFrame,
        inner: bool = True,
        encoder: ce.CatBoostEncoder = None,
        fit: bool = True
    ) -> Tuple[np.ndarray, pd.Series, ce.CatBoostEncoder]:
        """
        Prepare features for Task 2.

        Args:
            df: Input DataFrame
            inner: If True, prepare for inner model, else outer model
            encoder: Pre-fitted encoder (if fit=False)
            fit: Whether to fit the encoder

        Returns:
            Tuple of (feature array, target series, encoder)
        """
        df = df.copy()

        # Get numeric features
        num_features_to_cat = self.config.get('features.numeric_to_categorical')
        num_features = self.config.get('features.numeric_features')

        # Add composite features
        df = self.add_composite_features(df, task="task2")

        # Get target
        target = df['Target']

        # Encoding variables
        if inner:
            variables_to_encode = self.config.get('task2.inner.encoding_variables')
        else:
            variables_to_encode = self.config.get('task2.outer.encoding_variables')

        # Encode categorical features
        df_encoded, encoder = self.encode_categorical_features(
            df,
            variables_to_encode,
            target=target if fit else None,
            encoder=encoder,
            fit=fit,
            encoder_name=f"task2_{'inner' if inner else 'outer'}_encoder" if fit else None
        )

        # Select final features
        encoded_cols = [f"{x}_encoded" for x in variables_to_encode]
        count_features = ["Count_Pin", "Count_Pin_Figure"]

        if inner:
            # Inner model includes both numeric features and SPI features
            feature_cols = encoded_cols + num_features + num_features_to_cat + count_features
        else:
            # Outer model uses aggregated SPI features
            feature_cols = num_features_to_cat + encoded_cols + count_features

        X = df_encoded[feature_cols].fillna(0).to_numpy()

        logger.info(f"Task 2 ({'inner' if inner else 'outer'}) features prepared: {X.shape}")

        return X, target, encoder

    def prepare_features_task3(
        self,
        df: pd.DataFrame,
        encoder: ce.CatBoostEncoder = None,
        fit: bool = True,
        aoi_labels: List[str] = None
    ) -> Tuple[np.ndarray, pd.Series, ce.CatBoostEncoder, List[str]]:
        """
        Prepare features for Task 3.

        Args:
            df: Input DataFrame
            encoder: Pre-fitted encoder (if fit=False)
            fit: Whether to fit the encoder
            aoi_labels: List of AOI label names (if fit=False)

        Returns:
            Tuple of (feature array, target series, encoder, aoi_labels)
        """
        df = df.copy()

        # Get numeric features
        num_features_to_cat = self.config.get('features.numeric_to_categorical')
        num_features = self.config.get('features.numeric_features')

        # Add composite features
        df = self.add_composite_features(df, task="task3")

        # Create AOI label features
        if fit:
            df, aoi_labels = self.create_aoi_label_features(df)
        else:
            if aoi_labels is None:
                raise ValueError("aoi_labels is required when fit=False")
            # Initialize AOI label columns
            for label in aoi_labels:
                if label not in df.columns:
                    df[label] = 0

        # Get target
        target = df['Target']

        # Encoding variables
        variables_to_encode = self.config.get('task3.encoding_variables')

        # Encode categorical features
        df_encoded, encoder = self.encode_categorical_features(
            df,
            variables_to_encode,
            target=target if fit else None,
            encoder=encoder,
            fit=fit,
            encoder_name="task3_encoder" if fit else None
        )

        # Select final features
        encoded_cols = [f"{x}_encoded" for x in variables_to_encode]
        count_features = ["Count_Pin", "Count_Pin_Figure", "Count_Pin_Panel"]

        feature_cols = (
            num_features + encoded_cols + count_features +
            aoi_labels + num_features_to_cat + ["MachineID"]
        )

        X = df_encoded[feature_cols].fillna(0).to_numpy()

        logger.info(f"Task 3 features prepared: {X.shape}")

        return X, target, encoder, aoi_labels

    def save_encoder(self, encoder_name: str, file_path: str):
        """Save encoder to file."""
        if encoder_name not in self.encoders:
            raise ValueError(f"Encoder '{encoder_name}' not found")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.encoders[encoder_name], file_path)
        logger.info(f"Saved encoder '{encoder_name}' to {file_path}")

    @staticmethod
    def load_encoder(file_path: str) -> ce.CatBoostEncoder:
        """Load encoder from file."""
        encoder = joblib.load(file_path)
        logger.info(f"Loaded encoder from {file_path}")
        return encoder
