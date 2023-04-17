

from Model import SignLanguageModel
from main import *

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train(model, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()  # set the model to training mode

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")


if __name__ == '__main__':
    model = SignLanguageModel(500, 20)
    label_map = {}
    with open(LABEL_MAP_PATH) as fp:
        label_map = json.load(fp)
    actions = np.array(list(label_map.keys()))
    print("we have {} actions".format(actions.shape[0]))

    X_train, y_train = load_features(actions, label_map, data_type='test')
    X_test, y_test = load_features(actions, label_map, data_type='test')
    X_val, y_val = load_features(actions, label_map, data_type='test')

    # Convert the data to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()

    # print the shape of the data
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    # Create a PyTorch TensorDataset from your data
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    # Define batch size for training
    batch_size = 16
    # Create a train loader from the TensorDataset, with shuffle=True to randomly shuffle the data before each epoch
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define your optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, train_loader, val_loader, optimizer, criterion, 10)
