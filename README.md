# A Hierarchical XGBoost Early Detection Method for Quality and Productivity Improvement of Electronics Manufacturing Systems

This repository contains a clean, modular solution to the PHME Data Challenge 2022 for electronics manufacturing defect detection using XGBoost models.

**Challenge Links:**
- [Github Challenge](https://github.com/PHME-Datachallenge/Data-Challenge-2022)
- [Conference Challenge](https://phm-europe.org/data-challenge)

## 🎯 Overview

The solution addresses three classification tasks for electronics manufacturing quality control:

1. **Task 1**: Defect Detection from SPI (Solder Paste Inspection) data
2. **Task 2**: Operator Label Classification (Good/Bad) using SPI + AOI data
3. **Task 3**: Repair Label Classification (FalseScrap/NotPossibleToRepair)

## 🏗️ Project Structure

```
phme-data-contest/
├── config/
│   └── config.yaml              # Configuration for all tasks
├── src/
│   ├── data/
│   │   └── data_loader.py       # Data loading utilities
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering
│   ├── training/
│   │   └── trainer.py           # Training pipeline with MLflow
│   ├── inference/
│   │   └── predictor.py         # Inference pipeline
│   └── utils/
│       ├── config_loader.py     # Configuration management
│       └── logger.py            # Logging utilities
├── notebooks/
│   └── solution.ipynb           # Clean solution notebook
├── models/                      # Trained models (generated)
├── mlruns/                      # MLflow tracking data (generated)
├── data/                        # Training data
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/alexandregft/PHME-data-contest.git
cd PHME-data-contest
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data files are in place:**
```bash
ls data/
# Should show: AOI_training.csv.zip, SPI_training_*.csv.zip
```

## 🎓 Training Models

### Train All Tasks

```bash
python train.py --config config/config.yaml --task all
```

### Train Individual Tasks

```bash
# Train only Task 1
python train.py --task task1

# Train only Task 2
python train.py --task task2

# Train only Task 3
python train.py --task task3
```

### With Logging

```bash
python train.py --task all --log-file logs/training.log
```

## 📊 MLflow Experiment Tracking

The training pipeline integrates MLflow for experiment tracking. After training, view results:

```bash
mlflow ui
```

Then open your browser to `http://localhost:5000` to view:
- Hyperparameter optimization results
- Model performance metrics
- Training artifacts
- Model versioning

## 🔮 Making Predictions

### Using the Solution Notebook

The `notebooks/solution.ipynb` provides a clean interface for running predictions:

```python
from src.utils.config_loader import ConfigLoader
from src.inference.predictor import PHMEPredictor

# Initialize predictor
config = ConfigLoader('config/config.yaml')
predictor = PHMEPredictor(config)
predictor.load_models('models')

# Run predictions
defects = predictor.classification_1(spi_data)
operator_labels = predictor.classification_2(spi_data, aoi_data)
repair_labels = predictor.classification_3(spi_data, aoi_data)
```

### Using Pre-trained Models

Pre-trained models from the original solution are available in the `Training/` directory and can be used with the original notebooks.

## 📈 Model Architecture

### Task 1: Defect Detection
- **Input**: SPI features (volume, area, offset, shape, position, size)
- **Model**: XGBoost binary classifier
- **Output**: Defect detected (True/False) at component level

### Task 2: Operator Label Classification
- **Models**: Two XGBoost classifiers (inner/outer)
  - **Inner Model**: For components with specific pin numbers
  - **Outer Model**: For component-level defects (no specific pin)
- **Input**: SPI + AOI features, categorical encodings
- **Output**: Good/Bad classification

### Task 3: Repair Label Classification
- **Input**: SPI + AOI features + OperatorLabel
- **Model**: XGBoost classifier with AOI label one-hot encoding
- **Output**: FalseScrap/NotPossibleToRepair classification

## 🔧 Key Features

### Clean Architecture
- ✅ Modular code structure with clear separation of concerns
- ✅ Reusable components for data loading, feature engineering, and training
- ✅ Type hints and comprehensive documentation

### MLflow Integration
- ✅ Automatic experiment tracking
- ✅ Hyperparameter logging with Optuna
- ✅ Model registry and versioning
- ✅ Metric visualization

### Modern Best Practices
- ✅ Configuration management with YAML
- ✅ Structured logging
- ✅ Updated dependencies (XGBoost 1.7+, scikit-learn 1.2+)
- ✅ Command-line interface for training

## 📊 Performance

The solution achieves competitive performance on the PHME 2022 data challenge:
- **Task 1 F1 Score**: ~0.41
- **Task 2 F1 Score**: ~0.66
- **Task 3 F1 Score**: ~0.90
- **Overall Score**: ~0.66

## 🛠️ Configuration

Edit `config/config.yaml` to customize:
- Data paths
- MLflow settings
- Feature engineering parameters
- Hyperparameter search spaces
- Cross-validation settings

## 📝 Original Solution

The original Jupyter notebook-based solution is preserved in:
- `Training/Training_taskX.ipynb` - Training notebooks
- `Testing/solution.ipynb` - Original inference notebook

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🎓 Citation

If you use this code, please cite:

```bibtex
@misc{phme2022solution,
  author = {Alexandre GFT},
  title = {A Hierarchical XGBoost Early Detection Method for Electronics Manufacturing},
  year = {2022},
  publisher = {GitHub},
  url = {https://github.com/alexandregft/PHME-data-contest}
}
```

## 🙏 Acknowledgments

- PHME 2022 Data Challenge organizers
- The open-source community for the excellent tools (XGBoost, MLflow, Optuna)
