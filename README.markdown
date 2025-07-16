# SpatioTECformer: Ionospheric TEC Prediction Model

## Project Overview

SpatioTECformer is a deep learning model designed for predicting ionospheric Total Electron Content (TEC). It integrates multi-scale Convolutional Neural Networks (CNNs) with a Transformer architecture to process spatio-temporal data stored in HDF5 files. The model leverages multi-scale CNNs, adaptive feature fusion, and temporal encoding to forecast TEC values on a 71x73 latitude-longitude grid.

Implemented in Python using PyTorch, the project is modularized into five files to enhance maintainability and reusability.

## Project Structure

The project comprises five Python files, all located in the same directory:

- `main.py`: The primary script containing the training and testing logic, configuration, early stopping, and result-saving functions. It orchestrates the overall workflow.
- `data_utils.py`: Defines the `TECPredictDataset` class for data loading and preprocessing, along with functions for splitting and creating training, validation, and test dataloaders.
- `model.py`: Implements the `CNN_Encoder` class, which integrates CNNs, adaptive fusion, temporal encoding, and a Transformer encoder for TEC prediction.
- `temporal_encoding.py`: Defines two temporal encoding classes (`LearnableTemporalEncoding` and `TemporalEncoding`) to incorporate positional information into time series data.
- `enhanced_cnn.py`: Implements the `EnhancedCNN2D` class, which performs multi-scale (3x3, 5x5, 7x7) convolutional feature extraction with residual connections.
- `adaptive_fusion.py`: Contains the `AdaptiveFusion` class, which dynamically integrates CNN-extracted features with external features (e.g., temporal and auxiliary features) using a gating mechanism.

## Requirements

To run the project, the following dependencies must be installed:

- Python 3.11
- PyTorch (with CUDA support for GPU training)
- NumPy
- Pandas
- scikit-learn
- h5py
- tqdm
- tensorboard

Install the dependencies using:

```bash
pip install torch numpy pandas scikit-learn h5py tqdm tensorboard
```

## Setup

### Data Preparation

- The model requires an HDF5 file (`16_fused_with_features.h5`) containing TEC data, timestamps, latitude, longitude, and auxiliary features.
- Place the HDF5 file in the code directory, or update the `fused_hdf5_path` variable in `main.py` to match your data path.

### Directory Structure

Ensure all five Python files are located in the same directory. The expected structure is:

```
/
├── 16_fused_with_features.h5
├── main.py
├── data_utils.py
├── model.py
├── temporal_encoding.py
├── enhanced_cnn.py
├── adaptive_fusion.py
├── best_model.pth (generated during training)
├── SpatioTECformer_predictions8.h5 (generated during testing)
└── log/ (TensorBoard logs, created during training)
```

### Hardware Requirements

- The model supports multi-GPU training via `DataParallel` if GPUs are available.
- Ensure sufficient memory (RAM and GPU VRAM) to handle large HDF5 datasets and model training.

## Usage

### Running the Model

Execute the main script to train and test the model:

```bash
python main.py
```

- The script trains the model for up to 100 epochs, employing an early stopping mechanism (patience=30) based on validation loss.
- The best model weights are saved to `best_model.pth`.
- Upon completion of training, the model performs testing and saves predictions to `SpatioTECformer_predictions8.h5`.

### Monitoring Training

- Training and validation progress are logged using `logging` and visualized with TensorBoard.
- To view TensorBoard logs:

  ```bash
  tensorboard --logdir=log
  ```

  Open the provided URL in a browser.

### Output Results

- Test results are stored in HDF5 format (`SpatioTECformer_predictions8.h5`), containing sample IDs, timestamps, latitudes, longitudes, predicted values, and target values.
- The script outputs a summary of the HDF5 file contents and the first five entries for verification.

## Model Architecture

### Dataset (`TECPredictDataset`, in `data_utils.py`)

- Loads TEC data, latitude, longitude, and auxiliary features from an HDF5 file.
- Standardizes TEC data and encodes latitude/longitude as sine/cosine features.
- Extracts temporal features (month, day, weekday, hour, year, quarter) as auxiliary inputs.

### Temporal Encoding (`temporal_encoding.py`)

- Supports fixed (sinusoidal/cosinusoidal) or learnable positional encodings to capture temporal dependencies.

### Enhanced CNN (`enhanced_cnn.py`)

- Employs multi-scale convolutions (3x3, 5x5, 7x7) with residual connections to extract spatial features.

### Adaptive Fusion (`adaptive_fusion.py`)

- Dynamically integrates CNN-extracted features with external features (temporal and auxiliary) using a gating mechanism.

### CNN Encoder (`model.py`)

- Combines CNNs, adaptive fusion, temporal encoding, and an 8-layer Transformer encoder to predict TEC values for a 71x73 grid.

## Configuration Parameters

Key hyperparameters are defined in the `Configs` class in `main.py`:

- `seq_len`: 24 (input sequence length)
- `pred_len`: 1 (prediction length)
- `d_model`: 128 (model dimension)
- `n_heads`: 8 (Transformer attention heads)
- `dropout`: 0.2
- `batch_size`: 128
- `learning_rate`: 0.00005 (using AdamW optimizer)
- `scheduler`: ReduceLROnPlateau (factor=0.3, patience=30)
- `early_stopping`: Patience=30

Modify these parameters in `main.py` to conduct experiments.

## Notes

### Data Assumptions

- The HDF5 file must include `Timestamp`, `TEC`, `Latitude`, `Longitude`, and optional auxiliary features.
- Latitude and longitude grids are assumed to be fixed across all time steps.

### Error Handling

- Ensure the HDF5 file path is correct to avoid file-not-found errors.
- When using multiple GPUs, verify that all GPUs are accessible and compatible with PyTorch.

### Model Extension

- To incorporate new features, update the `TECPredictDataset` class in `data_utils.py` to include additional auxiliary features.
- To modify the model, adjust the `CNN_Encoder` class in `model.py` or add new modules in the respective files.

## Troubleshooting

### NameError: name 'torch' is not defined

- Ensure all modules (`data_utils.py`, `model.py`, `enhanced_cnn.py`, `temporal_encoding.py`, `adaptive_fusion.py`) include `import torch`.

### FileNotFoundError

- Verify the HDF5 file path in `main.py`.

### CUDA Out of Memory

- Reduce `batch_size` or disable `DataParallel` to use a single GPU.

## License

This project is intended for research purposes and is not distributed under a specific license. Contact the author for usage permissions.