import torch

class C3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(64),
            torch.nn.Mish(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(128),
            torch.nn.Mish(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            torch.nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.Mish(inplace=True),
            torch.nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.Mish(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 * 2 * 8 * 8, 512), # assume 8 x 64 x 64
            torch.nn.BatchNorm1d(512),
            torch.nn.Mish(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
        )

    def forward(self, x):
        h = self.cnn(x)
        h = torch.flatten(h, 1)
        h = self.fc(h)
        return h


class RNN(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.rnn = torch.nn.GRU(input_size=256, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.fc = torch.nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        logits = self.fc(out[:, -1, :])
        return logits


class C3D_RNN(torch.nn.Module):
    def __init__(self, clip_len=8):
        super().__init__()

        self.clip_len=clip_len

        self.c3d = C3D()
        self.rnn = RNN()

    def forward(self, video):
        _, _, video_len, _, _ = video.size()  # batch_size, channels, video_len, height, width

        clip_embs = []

        for i in range(0, video_len - (self.clip_len+1), self.clip_len):
            clip = video[:, :, i:i+self.clip_len, :, :]
            clip_emb = self.c3d(clip)
            clip_embs.append(clip_emb)

        clip_embs = torch.stack(clip_embs, dim=0).transpose(0, 1)
        logits = self.rnn(clip_embs)
        return logits
