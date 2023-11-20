import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt

# load the pickle file
with open('MindMNIST.pkl', 'rb') as f:
    arrays = pickle.load(f)
    z_arrays = pickle.load(f)
    digit_labels = pickle.load(f)

digit_labels = np.array(digit_labels)
arrays = np.stack(z_arrays, axis=0)
vec_length = arrays.shape[1]*arrays.shape[2]
arrays = arrays.reshape(-1,vec_length)

# test on 1 and 5
valid_data = (digit_labels == 1) | (digit_labels == 5)
X = arrays[valid_data, :]
y = digit_labels[valid_data]
y = y // np.max(y)
# breakpoint()

# digit_labels = np.array(digit_labels)
# valid_data = digit_labels >= 0
# arrays = np.stack(arrays, axis=0)[valid_data]
# digit_labels = digit_labels[valid_data]
# arrays = arrays.reshape(arrays.shape[0], -1)
# normalizer = 2500

# # Data Preparation
# np.random.seed(0)  # For reproducibility
# X = arrays/normalizer # 1000 samples of 500-dimensional vectors
# y = digit_labels  # Random labels for 10 classes
# # breakpoint()

# Convert to torch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.int64)

# Create a Dataset and DataLoader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model Definition
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(vec_length, vec_length//2),
            nn.ReLU(),
            nn.Linear(vec_length//2, 2)
        )
        # self.linear = nn.Linear(vec_length, 2)

    def forward(self, x):
        return self.linear(x)

model = LinearClassifier()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training Loop
losses = []
accs = []
for epoch in range(1000):  # number of epochs
    epoch_loss = 0
    epoch_acc = 0
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)

        probs = nn.Softmax()(outputs)
        clas = torch.max(probs, dim=1)[1]
        acc = torch.mean((clas == labels).float())
        # print(outputs[0])
        # print(labels[0])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_acc += acc.detach().item()
        # breakpoint()
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    losses.append(epoch_loss)
    accs.append(epoch_acc)

    print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Acc: {epoch_acc}')

plt.plot(losses)
plt.show()
