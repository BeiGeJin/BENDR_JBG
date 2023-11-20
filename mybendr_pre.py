from dn3_ext import ConvEncoderBENDR, BENDRContextualizer, BendingCollegeWav2Vec
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle

np.random.seed(0)  # For reproducibility

with open('MindMNIST.pkl', 'rb') as f:
    arrays = pickle.load(f) # list of numpy arrays, each numpy array correspond to an event after fft
    z_arrays = pickle.load(f)
    digit_labels = pickle.load(f) # list of digit labels corresponding to each event

# broadcast arrays
# arrays_processed = []
# for array in z_arrays:
#     array_4512 = np.zeros(shape=(4, 512))
#     array_4512[:,0:array.shape[1]] = array
#     array_20512 = np.tile(array_4512, (5,1))
#     arrays_processed.append(array_20512)
#     # breakpoint()

# Get rid of digit -1
# digit_labels = np.array(digit_labels)
# valid_data = digit_labels >= 0
# arrays = np.stack(arrays_processed, axis=0)[valid_data]
# digit_labels = digit_labels[valid_data]

# preprocess
arrays_processed = []
for array in z_arrays:
    array_20 = np.tile(array, (5,1))
    arrays_processed.append(array_20)

digit_labels = np.array(digit_labels)
arrays = np.stack(arrays_processed, axis=0)
# breakpoint()

# test on 1 and 5
valid_data = (digit_labels == 1) | (digit_labels == 5)
X = arrays[valid_data, :, :]
y = digit_labels[valid_data]
y = y // np.max(y)
# breakpoint()

# Convert to torch tensors
# X = arrays
# y = digit_labels
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.int64)

# Create a Dataset and DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define BENDR
encoder = ConvEncoderBENDR(20, encoder_h=512)
contextualizer = BENDRContextualizer(encoder.encoder_h)
encoder.load("pretrained/encoder.pt")
contextualizer.load("pretrained/contextualizer.pt")

# Run BENDR
vecs = []
digs = []
i = 0
for data in train_loader:
    if i % 10 == 0:
        print(i)
    i += 1

    inputs, labels = data
    encoded = encoder(inputs)
    context = contextualizer(encoded)
    vec = context[:,:,-1]
    vecs.append(vec.detach().numpy())
    digs.append(labels.detach().numpy())
    # breakpoint()

# save results
vecs_np = np.vstack(vecs)
digs_np = np.concatenate(digs)
with open('MindMNIST_bendr_pre.pkl', 'wb') as f:
    pickle.dump(vecs_np, f)
    pickle.dump(digs_np, f)

# testinput = torch.FloatTensor(np.random.random(size=(1, 20, 1536)))
# a, b, c = process.forward(testinput)
# print(process.forward(testinput))