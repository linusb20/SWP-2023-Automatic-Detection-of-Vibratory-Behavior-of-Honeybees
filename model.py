import torch

# N = Batch size
# L = Sequence Length
# C = Channels
# W = Width
# H = Height

# Input: N x L x C x W x H
# Output: N x L x H

class CNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, padding="same"),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 6, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
        )

    def forward(self, x):
        h_list = []
        num_frames = x.size(dim=1)

        for fi in range(num_frames):
            frame = x[:, fi, :, :, :]
            h = self.cnn(frame)
            h = torch.flatten(h, 1)
            h = self.fc(h)
            h_list.append(h)

        h_list = torch.stack(h_list, 0).transpose(0, 1)  # swap batch and sequence dimension
        return h_list

# N = Batch size
# L = Sequence Length
# H = Feature Size

# Input: N x L x H
# Output: N x H

class RNNDecoder(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        logits = self.fc(out[:, -1, :])
        return logits
