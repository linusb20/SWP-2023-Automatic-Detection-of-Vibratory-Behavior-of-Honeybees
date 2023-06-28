import torch
import torchvision.models as models
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
        resnet18 = models.resnet18(pretrained=True)
        resnet18_cnn = list(resnet18.children())[:-1]
        conv_weight = resnet18_cnn[0].weight
        resnet18_cnn[0].in_channels = 1
        resnet18_cnn[0].weight = torch.nn.Parameter(conv_weight.sum(dim=1, keepdim=True))

        self.cnn = torch.nn.Sequential(*resnet18_cnn)

        in_features = resnet18.fc.in_features

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
        )

    def forward(self, batch):
        video, video_lens = batch
        video_packed = pack_padded_sequence(video, video_lens.cpu(), batch_first=True, enforce_sorted=False) 

        with torch.no_grad():
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
