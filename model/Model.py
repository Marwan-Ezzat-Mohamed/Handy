import torch
import torch.nn as nn


import torch
import torch.nn as nn
from main import FRAMES_PER_VIDEO


class SignLanguageModel(nn.Module):
    def __init__(self, multi, actions):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv1d(126, 128*multi, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(128*multi)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(128*multi, 256*multi, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(256*multi)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Conv1d(256*multi, 512*multi, kernel_size=2)
        self.bn3 = nn.BatchNorm1d(512*multi)
        self.pool3 = nn.AvgPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(512*multi*(FRAMES_PER_VIDEO//8), 512*multi)
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512*multi, 256*multi)
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(256*multi, actions.shape[0])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1, 512*multi*(FRAMES_PER_VIDEO//8))

        x = self.fc1(x)
        x = nn.functional.elu(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        x = nn.functional.elu(x)
        x = self.dropout5(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x
