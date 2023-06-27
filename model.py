import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence 

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

    def forward(self, batch):
        video, video_lens = batch
        video_packed = pack_padded_sequence(video, video_lens.cpu(), batch_first=True, enforce_sorted=False) 
        x = self.cnn(video_packed.data) # video_packed.data contains all images of all videos in a batch
        x = torch.flatten(x, 1)
        h = self.fc(x)
        h_packed = PackedSequence(h, video_packed.batch_sizes, video_packed.sorted_indices, video_packed.unsorted_indices) # Replace images in PackedSequence with image embeddings
        return h_packed


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

    def forward(self, x_packed):
        self.rnn.flatten_parameters()
        out_packed, _ = self.rnn(x_packed)
        out_unpacked, out_lens = pad_packed_sequence(out_packed, batch_first=True)
        out = out_unpacked[torch.arange(out_unpacked.size(0)), out_lens - 1, :]
        logits = self.fc(out)
        return logits
