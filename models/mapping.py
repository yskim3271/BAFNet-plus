import torch
import math
from torch import nn
from torch.nn import functional as F
from torchaudio.models.conformer import ConformerLayer



class mapping(nn.Module):
    def __init__(self,
                 hidden=[64, 128, 256, 256, 256],
                 kernel_size=8,
                 stride=[2, 2, 4, 4, 4],
                 normalize=True,
                 rnn_layers=2
                 ):

        super().__init__()

        self.hidden = [1] + hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalize = normalize
        self.rnn_layers = rnn_layers

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        for index in range(len(self.hidden) - 1):
            encode = []
            encode += [
                nn.Conv1d(self.hidden[index], self.hidden[index + 1], kernel_size, self.stride[index]),
                nn.ReLU(),
                nn.Conv1d(self.hidden[index + 1], self.hidden[index + 1]* 2, 1), nn.GLU(1),
            ]
            self.encoder.append(nn.Sequential(*encode))
            
            decode = []
            decode += [
                nn.Conv1d(self.hidden[index + 1], self.hidden[index + 1]* 2, 1), nn.GLU(1), nn.ReLU(),
                nn.ConvTranspose1d(self.hidden[index + 1], self.hidden[index], kernel_size, self.stride[index]),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
        
        self.seq_encoder = nn.LSTM(self.hidden[-1], self.hidden[-1], self.rnn_layers, batch_first=True, bidirectional=False)


    def valid_length(self, length):
        for idx in range(len(self.encoder)):
            length = math.ceil((length - self.kernel_size) / self.stride[idx]) + 1
            length = max(length, 1)
        for idx in range(len(self.encoder)):
            length = (length - 1) * self.stride[idx] + self.kernel_size
        length = int(math.ceil(length))
        return int(length)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (1e-3 + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
            
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
                
        x = x.permute(0, 2, 1)
        x, _ = self.seq_encoder(x)
        x = x.permute(0, 2, 1)    
        
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)

        x = x[..., :length]
        return std * x


if __name__ == "__main__":
    model = mapping()
    x = torch.randn(2, 16000)
    y = model(x)
    print(y.shape)