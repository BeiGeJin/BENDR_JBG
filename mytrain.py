import torch
import tqdm
import argparse

from dn3_ext import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from dn3.transforms.batch import RandomTemporalCrop
from torch.utils.data import TensorDataset, DataLoader
import mne
import pickle
import numpy as np
mne.set_log_level(False)

# Dataset
# load the pickle file
with open('MindMNIST.pkl', 'rb') as f:
    arrays = pickle.load(f)
    z_arrays = pickle.load(f)
    digit_labels = pickle.load(f)

digit_labels = np.array(digit_labels)
arrays = np.stack(z_arrays, axis=0)

# test on 1 and 5
valid_data = (digit_labels == 1) | (digit_labels == 5)
X = arrays[valid_data, :]
y = digit_labels[valid_data]
y = y // np.max(y)
X = np.tile(X, 10)

# Convert to torch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.int64)

# Create a Dataset and DataLoader
training_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
encoder = ConvEncoderBENDR(4, encoder_h=512)
contextualizer = BENDRContextualizer(encoder.encoder_h)
process = BendingCollegeWav2Vec(encoder, contextualizer)

# Slower learning rate for the encoder
process.set_optimizer(torch.optim.Adam(process.parameters()))
process.add_batch_transform(RandomTemporalCrop())

process.fit(training_dataset, epochs=5, num_workers=0)
# print(process.evaluate(training_dataset))

tqdm.tqdm.write("Saving last model...")
encoder.save(f'checkpoints/my_encoder.pt')
contextualizer.save(f'checkpoints/my_contextualizer.pt')
