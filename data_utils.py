import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import h5py
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Dataset class (incorporates latitude and longitude sine/cosine encoding)
class TECPredictDataset(Dataset):
    def __init__(self, fused_hdf5_path, input_timestamps, output_timestamps, indices, scaler):
        self.fused_hdf5_path = fused_hdf5_path
        self.input_timestamps = input_timestamps
        self.output_timestamps = output_timestamps
        self.indices = indices
        self.scaler = scaler
        self.num_latitudes = 71
        self.num_longitudes = 73
        self.points_per_timestamp = self.num_latitudes * self.num_longitudes

        with h5py.File(fused_hdf5_path, 'r') as h5f:
            # Load timestamps and TEC data
            ts_bytes = h5f['Timestamp'][:]
            self.timestamps = pd.to_datetime([ts.decode('utf-8') for ts in ts_bytes],
                                             format='%Y-%m-%dT%H:%M:%S')
            T = len(self.timestamps)
            self.tec = h5f['TEC'][:].reshape(T, self.num_latitudes, self.num_longitudes)
            # Assume latitude and longitude grids are fixed across all time steps, using the first time step's data
            lat_grid = h5f['Latitude'][:].reshape(T, self.num_latitudes, self.num_longitudes)[0]
            lon_grid = h5f['Longitude'][:].reshape(T, self.num_latitudes, self.num_longitudes)[0]
            # Store original latitude and longitude grids (for result output)
            self.lat_grid = lat_grid
            self.lon_grid = lon_grid
            # Convert to radians and compute sine/cosine encodings
            self.lat_rad = np.deg2rad(lat_grid)
            self.lon_rad = np.deg2rad(lon_grid)
            self.lat_sin = np.sin(self.lat_rad)
            self.lat_cos = np.cos(self.lat_rad)
            self.lon_sin = np.sin(self.lon_rad)
            self.lon_cos = np.cos(self.lon_rad)

            max_idx = max(indices) if indices else 0
            if max_idx + input_timestamps + output_timestamps > T:
                raise ValueError(f"Index out of bounds: {max_idx} + {input_timestamps + output_timestamps} > {T}")

            # Extract temporal features (month, day, weekday, hour, year, quarter) as auxiliary inputs for the model
            time_df = pd.DataFrame({'Timestamp': self.timestamps})
            time_df['Month'] = time_df['Timestamp'].dt.month
            time_df['Day'] = time_df['Timestamp'].dt.day
            time_df['Weekday'] = time_df['Timestamp'].dt.weekday
            time_df['Hour'] = time_df['Timestamp'].dt.hour
            time_df['Year'] = time_df['Timestamp'].dt.year
            time_df['Quarter'] = time_df['Timestamp'].dt.quarter
            self.time_features = time_df[['Month', 'Day', 'Weekday', 'Hour', 'Year', 'Quarter']].values

            # Extract auxiliary features from the HDF5 file (physical quantities or features excluding Timestamp, Latitude, Longitude, TEC)
            self.aux_features = {key: h5f[key][:] for key in h5f.keys()
                                 if key not in ['Timestamp', 'Latitude', 'Longitude', 'TEC']}
            # Combine auxiliary features into an array; if no auxiliary features exist, use only temporal features
            aux_features_array = np.column_stack(list(self.aux_features.values())) if self.aux_features else None
            self.combined_features = np.hstack([self.time_features, aux_features_array]) \
                                     if aux_features_array is not None else self.time_features

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        mid_idx = start_idx + self.input_timestamps
        end_idx = mid_idx + self.output_timestamps

        inputs_tec = self.tec[start_idx:mid_idx]
        outputs = self.tec[mid_idx:end_idx]
        # Standardize input and output TEC data using a pre-trained StandardScaler
        inputs_tec_norm = self.scaler.transform(inputs_tec.reshape(-1, 1)).reshape(inputs_tec.shape)
        outputs_norm = self.scaler.transform(outputs.reshape(-1, 1)).reshape(outputs.shape)

        # Stack standardized TEC data with latitude and longitude sine/cosine encodings to form 5 channels (TEC, lat_sin, lat_cos, lon_sin, lon_cos)
        lat_sin = np.repeat(self.lat_sin[np.newaxis, :, :], self.input_timestamps, axis=0)
        lat_cos = np.repeat(self.lat_cos[np.newaxis, :, :], self.input_timestamps, axis=0)
        lon_sin = np.repeat(self.lon_sin[np.newaxis, :, :], self.input_timestamps, axis=0)
        lon_cos = np.repeat(self.lon_cos[np.newaxis, :, :], self.input_timestamps, axis=0)
        inputs = np.stack([inputs_tec_norm, lat_sin, lat_cos, lon_sin, lon_cos], axis=1)

        input_marks = self.combined_features[start_idx:mid_idx]
        output_marks = self.combined_features[mid_idx:end_idx]

        return (torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(outputs_norm, dtype=torch.float32),
                torch.tensor(input_marks, dtype=torch.float32),
                torch.tensor(output_marks, dtype=torch.float32))

# Dataset splitting and DataLoader creation
def prepare_dataloaders(fused_hdf5_path, input_timestamps=24, output_timestamps=1, stride=1, batch_size=128):
    with h5py.File(fused_hdf5_path, 'r') as h5f:
        T = h5f['Timestamp'].shape[0]
        tec_data_all = h5f['TEC'][:].reshape(T, 71, 73)
    window_size = input_timestamps + output_timestamps
    num_windows = (T - window_size) // stride + 1
    all_indices = [i * stride for i in range(num_windows)]
    logger.info(f"Number of sliding windows generated: {num_windows}")

    total = num_windows
    train_end1 = int(0.375 * total)
    train_start2 = int(0.4375 * total)
    train_end2 = int(0.5625 * total)
    val_start = int(0.625 * total)

    train_indices = all_indices[:train_end1] + all_indices[train_start2:train_end2]
    val_indices = all_indices[val_start:]
    test_indices = all_indices[train_end1:train_start2] + all_indices[train_end2:val_start]
    logger.info(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    scaler_tec = StandardScaler()
    tec_flat = tec_data_all.reshape(-1, 1)
    scaler_tec.fit(tec_flat)
    logger.info("Data preprocessing completed, StandardScaler fitted")

    train_dataset = TECPredictDataset(fused_hdf5_path, input_timestamps, output_timestamps, train_indices, scaler_tec)
    val_dataset = TECPredictDataset(fused_hdf5_path, input_timestamps, output_timestamps, val_indices, scaler_tec)
    test_dataset = TECPredictDataset(fused_hdf5_path, input_timestamps, output_timestamps, test_indices, scaler_tec)

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)
    logger.info("Data loading completed.")

    return train_loader, val_loader, test_loader, test_dataset, test_indices