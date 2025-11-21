"""Data loading utilities for PHME data challenge."""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare SPI and AOI data."""

    def __init__(self, config):
        """
        Initialize data loader.

        Args:
            config: Configuration object
        """
        self.config = config

    def load_spi_data(self, file_pattern: str = None, num_files: int = None) -> pd.DataFrame:
        """
        Load SPI training data from multiple CSV files.

        Args:
            file_pattern: Pattern for SPI file paths (e.g., 'data/SPI_training_{}.csv.zip')
            num_files: Number of SPI files to load

        Returns:
            Concatenated SPI DataFrame
        """
        if file_pattern is None:
            file_pattern = self.config.get('data.spi_training_pattern')
        if num_files is None:
            num_files = self.config.get('data.num_spi_files')

        logger.info(f"Loading {num_files} SPI data files...")

        spi_dfs = []
        for i in range(num_files):
            file_path = file_pattern.format(i)
            if not Path(file_path).exists():
                logger.warning(f"SPI file not found: {file_path}")
                continue

            logger.info(f"Loading {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            spi_dfs.append(df)

        if not spi_dfs:
            raise FileNotFoundError("No SPI data files found")

        spi_data = pd.concat(spi_dfs, ignore_index=True)
        logger.info(f"Loaded SPI data: {spi_data.shape[0]} rows, {spi_data.shape[1]} columns")

        return spi_data

    def load_aoi_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load AOI training data.

        Args:
            file_path: Path to AOI CSV file

        Returns:
            AOI DataFrame
        """
        if file_path is None:
            file_path = self.config.get('data.aoi_training')

        if not Path(file_path).exists():
            raise FileNotFoundError(f"AOI file not found: {file_path}")

        logger.info(f"Loading AOI data from {file_path}")
        aoi_data = pd.read_csv(file_path)
        logger.info(f"Loaded AOI data: {aoi_data.shape[0]} rows, {aoi_data.shape[1]} columns")

        return aoi_data

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both SPI and AOI data.

        Returns:
            Tuple of (SPI DataFrame, AOI DataFrame)
        """
        spi_data = self.load_spi_data()
        aoi_data = self.load_aoi_data()

        return spi_data, aoi_data

    @staticmethod
    def prepare_spi_for_task1(spi_data: pd.DataFrame, aoi_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare SPI data for Task 1 (defect detection).

        Args:
            spi_data: Raw SPI DataFrame
            aoi_data: Raw AOI DataFrame

        Returns:
            Prepared SPI DataFrame with target
        """
        logger.info("Preparing data for Task 1...")

        spi = spi_data.copy()

        # Convert PinNumber to string for merging
        spi['PinNumber'] = spi['PinNumber'].astype(str)

        # Create ID copies for merging
        spi["ComponentID2"] = spi["ComponentID"]
        spi["FigureID2"] = spi["FigureID"]

        # Prepare AOI data
        aoi = aoi_data.copy()
        aoi['PinNumber'] = aoi['PinNumber'].astype('Int64').astype(str)

        # Merge SPI with AOI
        spi_merged = spi.merge(
            aoi,
            on=['PanelID', 'FigureID', 'ComponentID', 'PinNumber'],
            how="left"
        )
        spi_merged = spi_merged.drop_duplicates(
            subset=['PanelID', 'FigureID', 'ComponentID', 'PinNumber']
        )

        # Create target: True if MachineID is NA (defect detected)
        spi_merged['Target'] = spi_merged['MachineID'].isna()

        logger.info(f"Task 1 data prepared: {spi_merged.shape[0]} rows")
        logger.info(f"Target distribution: {spi_merged['Target'].value_counts().to_dict()}")

        return spi_merged

    @staticmethod
    def prepare_data_for_task2(
        spi_data: pd.DataFrame,
        aoi_data: pd.DataFrame,
        inner: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for Task 2 (operator label classification).

        Args:
            spi_data: Raw SPI DataFrame
            aoi_data: Raw AOI DataFrame
            inner: If True, prepare for inner model (with PinNumber), else outer model

        Returns:
            Prepared DataFrame with features and target
        """
        logger.info(f"Preparing data for Task 2 ({'inner' if inner else 'outer'} model)...")

        # Prepare AOI data
        aoi = aoi_data.copy()
        aoi['PinNumber'] = aoi['PinNumber'].astype('Int64').astype(str)
        aoi['FigureID2'] = aoi['FigureID']
        aoi['ComponentID2'] = aoi['ComponentID']

        # Add count features
        aoi['Count_Pin'] = aoi.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )["PinNumber"].transform('count')
        aoi['Count_Pin_Figure'] = aoi.groupby(
            ["PanelID", "FigureID2"]
        )["PinNumber"].transform('count')
        aoi['Count_Pin_Panel'] = aoi.groupby(
            ["PanelID"]
        )["PinNumber"].transform('count')

        # Create target
        aoi['Target'] = aoi['OperatorLabel'].astype('category').cat.codes

        # Filter based on model type
        if inner:
            # Inner model: rows with actual PinNumber
            aoi = aoi.loc[aoi['PinNumber'] != '<NA>']
        else:
            # Outer model: rows without PinNumber
            aoi = aoi.loc[aoi['PinNumber'] == '<NA>']

        # Prepare SPI data
        spi = spi_data.copy()
        spi = spi.dropna(subset=['FigureID'])
        spi['FigureID2'] = spi['FigureID']
        spi['ComponentID2'] = spi['ComponentID']

        if inner:
            # Inner model: merge on PinNumber
            spi['PinNumber'] = spi['PinNumber'].astype(str)
            merged = aoi.merge(
                spi,
                on=['PanelID', 'FigureID', 'ComponentID', 'PinNumber'],
                how="inner"
            )
        else:
            # Outer model: aggregate SPI features at component level
            features_to_agg = ['Volume(%)', 'Area(%)', 'OffsetX(%)', 'OffsetY(%)']
            spi_grouped = spi.groupby(
                ["PanelID", "FigureID2", "ComponentID2"]
            )[features_to_agg].mean().reset_index()

            merged = aoi.merge(
                spi_grouped,
                on=['PanelID', 'FigureID2', 'ComponentID2'],
                how="left"
            )

        logger.info(f"Task 2 ({'inner' if inner else 'outer'}) data prepared: {merged.shape[0]} rows")
        logger.info(f"Target distribution: {merged['Target'].value_counts().to_dict()}")

        return merged

    @staticmethod
    def prepare_data_for_task3(spi_data: pd.DataFrame, aoi_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Task 3 (repair label classification).

        Args:
            spi_data: Raw SPI DataFrame
            aoi_data: Raw AOI DataFrame

        Returns:
            Prepared DataFrame with features and target
        """
        logger.info("Preparing data for Task 3...")

        # Prepare AOI data
        aoi = aoi_data.copy()
        aoi = aoi.loc[aoi['OperatorLabel'] == 'Bad']  # Only bad components need repair classification

        aoi['PinNumber'] = aoi['PinNumber'].astype('Int64').astype(str)
        aoi['FigureID2'] = aoi['FigureID']
        aoi['ComponentID2'] = aoi['ComponentID']

        # Add count features
        aoi['Count_Pin'] = aoi.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )["PinNumber"].transform('count')
        aoi['Count_Pin_Figure'] = aoi.groupby(
            ["PanelID", "FigureID2"]
        )["PinNumber"].transform('count')
        aoi['Count_Pin_Panel'] = aoi.groupby(
            ["PanelID"]
        )["PinNumber"].transform('count')

        # Create target (FalseScrap=0, NotPossibleToRepair=1)
        aoi['Target'] = aoi['RepairLabel'].map({
            'FalseScrap': 0,
            'NotPossibleToRepair': 1
        })

        # Filter valid targets
        aoi = aoi.dropna(subset=['Target'])

        # Machine ID encoding
        aoi['MachineID'] = aoi['MachineID'].astype('category').cat.codes

        # Create key for merging
        aoi['key_spi'] = (
            aoi['PanelID'].astype(str) + '_' +
            aoi['FigureID'].astype(str) + '_' +
            aoi['ComponentID'].astype(str)
        )

        # Prepare SPI data
        spi = spi_data.copy()
        spi = spi.dropna(subset=['FigureID'])
        spi['FigureID2'] = spi['FigureID']
        spi['ComponentID2'] = spi['ComponentID']
        spi['key_spi'] = (
            spi['PanelID'].astype(str) + '_' +
            spi['FigureID'].astype(int, errors='ignore').astype(str) + '_' +
            spi['ComponentID'].astype(str)
        )

        # Filter SPI to only components in AOI
        spi = spi.loc[spi['key_spi'].isin(aoi['key_spi'].unique())]

        # Aggregate SPI features at component level
        features_to_agg = [
            'Volume(%)', 'Area(%)', 'OffsetX(%)', 'OffsetY(%)',
            'Shape(um)', 'PosX(mm)', 'PosY(mm)', 'SizeX', 'SizeY'
        ]
        spi['Shape(um)'] = pd.to_numeric(spi['Shape(um)'], errors='coerce')
        spi_grouped = spi.groupby(
            ["PanelID", "FigureID2", "ComponentID2"]
        )[features_to_agg].mean().reset_index()

        # Merge with AOI
        merged = aoi.merge(
            spi_grouped,
            on=['PanelID', 'FigureID2', 'ComponentID2'],
            how="left"
        )

        # Drop duplicates at component level
        merged = merged.drop_duplicates(subset=['PanelID', 'FigureID2', 'ComponentID2'])

        logger.info(f"Task 3 data prepared: {merged.shape[0]} rows")
        logger.info(f"Target distribution: {merged['Target'].value_counts().to_dict()}")

        return merged
