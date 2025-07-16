import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import h5py
import logging
from data_utils import prepare_dataloaders
from model import CNN_Encoder

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Set random seed for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")

# Configuration section
fused_hdf5_path = '16_fused_with_features.h5'
with h5py.File(fused_hdf5_path, 'r') as h5f:
    aux_keys = [k for k in h5f.keys() if k not in ['Timestamp', 'Latitude', 'Longitude', 'TEC']]
    num_aux_features = len(aux_keys)

class Configs:
    def __init__(self, num_aux_features):
        self.seq_len = 24
        self.label_len = 24  # Used to construct decoder input (historical part)
        self.pred_len = 1
        self.d_model = 128
        self.enc_in = 128
        self.dec_in = 128
        self.c_out = 128
        self.d_ff = 128
        self.n_heads = 8
        self.d_layers = 8  # Number of Transformer encoder layers currently used
        self.dropout = 0.2
        self.activation = "relu"
        self.output_attention = False
        self.embed = 'fixed'
        self.freq = 'h'
        # Dimension of auxiliary features
        self.input_feature_dim = 6 + num_aux_features

configs = Configs(num_aux_features=num_aux_features)
cnn_channels = 48
cnn_kernel_size = 3

# EarlyStopping implementation
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict())
            self.counter = 0

# Result saving functions (CSV and HDF5)
def save_results_to_csv(predictions, targets, test_indices, input_timestamps,
                        timestamps, lat_array, lon_array, output_csv_path):
    rows = []
    num_samples, pred_len, lat, lon = predictions.shape
    for s in range(num_samples):
        start = test_indices[s] + input_timestamps
        for t in range(pred_len):
            real_timestamp = timestamps[start + t]
            for i in range(lat):
                for j in range(lon):
                    rows.append({
                        'sample': s,
                        'timestamp': real_timestamp,
                        'latitude': lat_array[start + t, i, j],
                        'longitude': lon_array[start + t, i, j],
                        'prediction': predictions[s, t, i, j],
                        'target': targets[s, t, i, j]
                    })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to CSV file: {output_csv_path}")

def save_results_to_hdf5(predictions, targets, test_indices, input_timestamps,
                         timestamps, lat_array, lon_array, output_hdf5_path):
    num_samples, pred_len, lat, lon = predictions.shape
    total_entries = num_samples * pred_len * lat * lon
    sample_ids = np.zeros(total_entries, dtype=np.int32)
    timestamp_data = np.empty(total_entries, dtype='datetime64[s]')
    latitudes = np.zeros(total_entries, dtype=np.float32)
    longitudes = np.zeros(total_entries, dtype=np.float32)
    pred_values = np.zeros(total_entries, dtype=np.float32)
    target_values = np.zeros(total_entries, dtype=np.float32)
    idx = 0
    np_timestamps = np.array(timestamps, dtype='datetime64[s]')

    for s in range(num_samples):
        start = test_indices[s] + input_timestamps
        for t in range(pred_len):
            real_timestamp = np_timestamps[start + t]
            for i in range(lat):
                for j in range(lon):
                    sample_ids[idx] = s
                    timestamp_data[idx] = real_timestamp
                    latitudes[idx] = lat_array[start + t, i, j]
                    longitudes[idx] = lon_array[start + t, i, j]
                    pred_values[idx] = predictions[s, t, i, j]
                    target_values[idx] = targets[s, t, i, j]
                    idx += 1
    with h5py.File(output_hdf5_path, 'w') as h5f:
        h5f.create_dataset('sample', data=sample_ids, compression='gzip', compression_opts=4)
        h5f.create_dataset('timestamp', data=timestamp_data.astype('S'), compression='gzip', compression_opts=4)
        h5f.create_dataset('latitude', data=latitudes, compression='gzip', compression_opts=4)
        h5f.create_dataset('longitude', data=longitudes, compression='gzip', compression_opts=4)
        h5f.create_dataset('prediction', data=pred_values, compression='gzip', compression_opts=4)
        h5f.create_dataset('target', data=target_values, compression='gzip', compression_opts=4)
    logger.info(f"Results saved to HDF5 file: {output_hdf5_path}")

# Training and testing procedure
if __name__ == '__main__':
    set_random_seed(seed=42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    input_timestamps = 24
    output_timestamps = 1
    stride = 1
    batch_size = 128
    train_loader, val_loader, test_loader, test_dataset, test_indices = prepare_dataloaders(
        fused_hdf5_path, input_timestamps, output_timestamps, stride, batch_size
    )

    # Initialize the model
    model = CNN_Encoder(
        configs=configs,
        cnn_channels=cnn_channels,
        cnn_kernel_size=cnn_kernel_size,
        num_latitudes=71,
        num_longitudes=73,
        input_feature_dim=configs.input_feature_dim,
        use_learnable_temporal=True
    ).to(device)
    logger.info("Model initialization completed, architecture: CNN_Encoder with Transformer")

    if torch.cuda.device_count() > 1:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel.")
        model = torch.nn.DataParallel(model)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-5)
    # Initialize learning rate scheduler: Reduces learning rate by a factor of 0.3 if validation loss does not improve for 30 epochs
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30)
    # Initialize early stopping: Halts training if validation loss does not improve for 30 epochs
    early_stopping = EarlyStopping(patience=30, verbose=True)
    writer = SummaryWriter(log_dir='log')

    logger.info("Starting model training...")
    epochs = 100
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}", leave=True) as pbar:
            for data, target, input_mark, output_mark in train_loader:
                x_dec = data  # (batch_size, 24, 5, 71, 73) - historical data only
                x_mark_dec = input_mark  # (batch_size, 24, input_feature_dim)
                x_dec = x_dec.to(device)
                x_mark_dec = x_mark_dec.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                outputs_pred = model(x_dec, x_mark_dec)
                loss = criterion(outputs_pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))
                pbar.update(1)
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}")
        writer.add_scalar('Train/Loss', avg_train_loss, epoch + 1)
        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0
        with tqdm(total=len(val_loader), desc=f"Validation Epoch {epoch+1}/{epochs}", leave=True) as val_pbar:
            with torch.no_grad():
                for data, target, input_mark, output_mark in val_loader:
                    x_dec = data  # (batch_size, 24, 5, 71, 73)
                    x_mark_dec = input_mark  # (batch_size, 24, input_feature_dim)
                    x_dec = x_dec.to(device)
                    x_mark_dec = x_mark_dec.to(device)
                    target = target.to(device)
                    outputs_pred = model(x_dec, x_mark_dec)
                    loss = criterion(outputs_pred, target)
                    val_loss += loss.item()
                    val_pbar.set_postfix(loss=val_loss / (val_pbar.n + 1))
                    val_pbar.update(1)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.6f}")
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch + 1)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch + 1)
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), 'best_model.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')
            logger.info("Best model saved!")
            writer.add_scalar('Best_Validation_Loss', best_val_loss, epoch + 1)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered, halting training!")
            break
    writer.close()

    logger.info("Starting testing...")
    best_model_path = 'best_model.pth'
    state_dict = torch.load(best_model_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        if not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
    else:
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Best model weights loaded successfully!")

    test_loss = 0.0
    all_preds_inverse = []
    all_targets_inverse = []
    with torch.no_grad():
        for data, target, input_mark, output_mark in tqdm(test_loader, desc="Testing"):
            x_dec = data  # (batch_size, 24, 5, 71, 73)
            x_mark_dec = input_mark  # (batch_size, 24, input_feature_dim)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            target = target.to(device)
            outputs_pred = model(x_dec, x_mark_dec)
            loss = criterion(outputs_pred, target)
            test_loss += loss.item()
            outputs_pred_np = outputs_pred.cpu().numpy().reshape(-1, 71 * 73)
            targets_np = target.cpu().numpy().reshape(-1, 71 * 73)
            outputs_pred_inverse = test_dataset.scaler.inverse_transform(outputs_pred_np)
            targets_inverse = test_dataset.scaler.inverse_transform(targets_np)
            all_preds_inverse.append(outputs_pred_inverse.reshape(-1, configs.pred_len, 71, 73))
            all_targets_inverse.append(targets_inverse.reshape(-1, configs.pred_len, 71, 73))
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Average test loss: {avg_test_loss:.6f}")
    all_preds_inverse = np.concatenate(all_preds_inverse, axis=0)
    all_targets_inverse = np.concatenate(all_targets_inverse, axis=0)

    # Use original latitude and longitude grids for result storage, extending lat_grid and lon_grid to match the timestamp length for alignment with predictions
    original_lat = test_dataset.lat_grid[np.newaxis, :, :].repeat(len(test_dataset.timestamps), axis=0)
    original_lon = test_dataset.lon_grid[np.newaxis, :, :].repeat(len(test_dataset.timestamps), axis=0)

    save_results_to_hdf5(
        predictions=all_preds_inverse,
        targets=all_targets_inverse,
        test_indices=test_indices,
        input_timestamps=input_timestamps,
        timestamps=test_dataset.timestamps,
        lat_array=original_lat,
        lon_array=original_lon,
        output_hdf5_path='SpatioTECformer_predictions8.h5'
    )
    with h5py.File('SpatioTECformer_predictions8.h5', 'r') as h5f:
        print("HDF5 file contents:")
        for key in h5f.keys():
            print(f"{key}: {h5f[key].shape}, dtype: {h5f[key].dtype}")
        sample = h5f['sample'][:5]
        timestamp = pd.to_datetime(h5f['timestamp'][:5].astype(str))
        latitude = h5f['latitude'][:5]
        longitude = h5f['longitude'][:5]
        prediction = h5f['prediction'][:5]
        target = h5f['target'][:5]
        for i in range(5):
            print(f"Sample {sample[i]}, Time: {timestamp[i]}, Lat: {latitude[i]}, Lon: {longitude[i]}, Pred: {prediction[i]}, Target: {target[i]}")