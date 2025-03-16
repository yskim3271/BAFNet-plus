import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module_dccrn import ConvSTFT, ConviSTFT, \
    ComplexConv2d, ComplexConvTranspose2d, complex_cat, NavieComplexLSTM

class dccrn_plus(nn.Module):

    def __init__(
            self,
            ctcnn_layers=4,
            ctcnn_kernel_size=5,
            win_len=400,
            win_inc=100,
            fft_len=512,
            win_type='hann',
            kernel_size=5,
            kernel_num=[16, 32, 64, 128, 256, 256],
    ):
        '''

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(dccrn_plus, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.ctcnn_layers = ctcnn_layers
        self.ctcnn_kernel_size = ctcnn_kernel_size

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.kernel_num = [2]  + kernel_num

        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
          
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        
            
        self.ctcnn = nn.ModuleList()
        
        for idx in range(ctcnn_layers):
            dialation = 2 ** idx
            self.ctcnn.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[-1],
                        self.kernel_num[-1],
                        kernel_size= (3, ctcnn_kernel_size),
                        stride=(1, 1),
                        padding=(1, (ctcnn_kernel_size-1) * dialation),
                        dilation=(1, dialation)
                    ),
                    nn.BatchNorm2d(self.kernel_num[-1]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                    )
                )
            

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                    )
                )


    def forward(self, inputs, lens=False):
        
        in_len = inputs.size(-1)
        
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, imag], 1)
        
        cspecs = cspecs[:, :, 1:]
        
        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)
                
        for idx in range(len(self.ctcnn)):
            out = self.ctcnn[idx](out) + out
            
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
                
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )

        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.clamp_(out_wav, -1, 1)

        out_len = out_wav.size(-1)
        
        if out_len > in_len:
            leftover = out_len - in_len 
            out_wav = out_wav[..., leftover//2:-(leftover//2)]
                
        if lens == True:
            return mask_mags, out_spec, out_wav
        else:
            return out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params


if __name__ == "__main__":
    input =  torch.randn(1, 1600)
    
    model = dccrn_plus(
        rnn_layers=2,
        rnn_units=256,
        use_clstm=True,
        kernel_size=5,
        kernel_num=[32, 64, 128, 256, 256, 256],
        win_type='hann',
        win_len=400,
        win_inc=100,
        fft_len=512,
        num_k=4
    )
    output = model(input)
    print(output.shape)
    print(output)
