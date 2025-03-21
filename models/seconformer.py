import torch
import math
from torch import nn
from torch.nn import functional as F
from torchaudio.models.conformer import ConformerLayer, _FeedForwardModule, _ConvolutionModule
import copy

def sinc(t):
    """sinc.

    :param t: the input tensor
    """
    return torch.where(t == 0, torch.tensor(1., device=t.device, dtype=t.dtype), torch.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.

    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = torch.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.

    """
    win = torch.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = torch.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)

def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    '''
    
    query, key, value : All are projections of input
    mask : To ensure future words are unreachable

    '''
    B,_,T,d_k = query.size()
    scores = torch.matmul(query,key.transpose(-2,-1))/(math.sqrt(d_k)) # dot product b/w query,key

    if mask is not None:
        scores = scores.masked_fill(mask[:,:,:T,:T]==0,-1e10) # make future words unreachable -inf
    prob_attn = F.softmax(scores,dim=-1) # calculating probs
    if dropout is not None:
        prob_attn = dropout(prob_attn) # pass through dropout

    return torch.matmul(prob_attn,value) # attn_weights * value # weighted sum of values. each emb idx is a weighted sum of all other emb idxs of all T values

class SynthesizerAttention(nn.Module):
    def __init__(self, n_embd,n_head,block_size,attn_pdrop,resid_pdrop):
        """
        
        n_embd : embedding size 
        n_head : number of attention heads
        block_size : length of seq
        attn_pdrop : attention dropout probability
        resid_pdrop : dropout prob after projection layer.
        
        """
        super().__init__()
        assert n_embd %n_head == 0
        self.w1 = nn.Linear(n_embd, n_embd)
        self.w2 = nn.Parameter(torch.zeros( n_embd //  n_head,
             block_size-1)) #d_k,T
        self.b2 = nn.Parameter(torch.zeros(block_size-1)) #T
        # value projection
        self.value = nn.Linear( n_embd,  n_embd) #dmodel,dmodel
        # regularization
        self.attn_drop = nn.Dropout( attn_pdrop)
        self.resid_drop = nn.Dropout( resid_pdrop)
        # output projection
        self.proj = nn.Linear( n_embd,  n_embd) #dmodel,dmodel
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones( block_size,  block_size)).view(
                1, 1,  block_size,  block_size)) #mask
        self.n_head =  n_head
        self.block_size =  block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # @ : The matrix multiplication(s) are done between the last two dimensions
        d_k = C//self.n_head
        relu_out = F.relu(self.w1(x)).\
            view(B,T,self.n_head,d_k).transpose(1,2)     

        v = self.value(x).view(B,T,self.n_head,d_k).transpose(1,2)   
        scores = (relu_out@self.w2)  + self.b2  
        
        scores = scores[:,:,:T,:T] # to ensure it runs for T<block_size
        scores = scores.masked_fill(self.mask[:,:,:T,:T]==0,-1e10)
        prob_attn = F.softmax(scores,dim=-1)
        y = prob_attn@v

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_drop(self.proj(y))
        return y

class CausalSelfAttention(nn.Module):

    ''' 
        n_embd : embedding size 
        n_head : number of attention heads
        block_size : length of seq
        attn_pdrop : attention dropout probability
        resid_pdrop : dropout prob after projection layer.
        
    '''
    def __init__(self, n_embd,n_head,attn_pdrop,resid_pdrop,block_size):
        super().__init__()
        d_model =  n_embd
        self.n_head =  n_head
        assert d_model %  n_head == 0 # d_model/n_head are divisble
        self.d_k = d_model//self.n_head

        self.linears = clones(nn.Linear(d_model,d_model),4) # key, value, query, out_proj
        
        self.attn_drop = nn.Dropout( attn_pdrop)
        self.resid_drop = nn.Dropout( resid_pdrop)

        block_size =  block_size
        # to hide future words
        subsequent_mask = torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size) # original
        # subsequent_mask = torch.triu(torch.ones(block_size,block_size)).view(1,1,block_size,block_size)
        self.register_buffer("mask",subsequent_mask) # to make sure it is stored in states dict while saving model

      
    def forward(self, x, layer_past=None):
        B,T,d_model = x.size()
        query,key,value = [l(x).view(B,-1,self.n_head,self.d_k).transpose(1,2) for l,x in zip(self.linears,(x,x,x))]
        #print(x.shape)
        y = attention(query,key,value,mask=self.mask,dropout=self.attn_drop)
        
        y = y.transpose(1,2).contiguous().view(B,T,d_model)

        return self.resid_drop(self.linears[-1](y)) #pass through a linear and dropout


class CustomConformerLayer(nn.Module):
    """확장된 ConformerLayer 구현.
    
    torchaudio의 ConformerLayer를 기반으로 하지만, causal 속성을 설정할 수 있는 확장된 버전.
    """
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        # self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        # if is_causal:
        #     self.self_attn = CausalSelfAttention(
        #         n_embd=input_dim,
        #         n_head=num_attention_heads,
        #         attn_pdrop=dropout,
        #         resid_pdrop=dropout,
        #         block_size=256,
        #     )
        #     # self.self_attn = SynthesizerAttention(
        #     #     n_embd=input_dim,
        #     #     n_head=num_attention_heads,
        #     #     block_size=256,
        #     #     attn_pdrop=dropout,
        #     #     resid_pdrop=dropout,
        #     # )
        # else:
        #     self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        # self.self_attn_dropout = torch.nn.Dropout(dropout)
        # self.is_causal = is_causal

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual
        
        if self.convolution_first:
            x = self._apply_convolution(x)
        
        # residual = x
        # x = self.self_attn_layer_norm(x)
        
        
        # if self.is_causal:
        #     x = self.self_attn(x)
        # else:
        #     x, _ = self.self_attn(x, x, x)
        
        # x = self.self_attn_dropout(x)
        # x = x + residual
        
        if not self.convolution_first:
            x = self._apply_convolution(x)
        
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        
        x = self.final_layer_norm(x)
        return x
            



class seconformer(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=32,
                 depth=4,
                 conformer_dim=256,
                 conformer_ffn_dim=256,
                 conformer_num_attention_heads=4,
                 conformer_depth = 2,
                 depthwise_conv_kernel_size=31,
                 kernel_size=8,
                 stride=4,
                 resample=4,
                 growth=2,
                 dropout=0.1,
                 rescale=0.1,
                 normalize=True,
                 sample_rate=16_000,
                 causal=False):

        super().__init__()

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.resample = resample
        self.growth = growth
        self.dropout = dropout
        self.conformer_dim = conformer_dim
        self.conformer_ffn_dim = conformer_ffn_dim
        self.conformer_num_attention_heads = conformer_num_attention_heads
        self.conformer_depth = conformer_depth
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.causal = causal

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ch_scale = 2 

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), nn.GLU(1),
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), nn.GLU(1), nn.ReLU(),
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = int(growth * hidden)

        self.conformers = nn.ModuleList()
        
        for index in range(conformer_depth):
            self.conformers.append(
                CustomConformerLayer(
                    input_dim=conformer_dim,
                    ffn_dim=conformer_ffn_dim,
                    num_attention_heads=conformer_num_attention_heads,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    is_causal=causal,
                )
            )
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
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

        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
            
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        
        x = x.permute(2, 0, 1)
        
        for conformer in self.conformers:
            x = conformer(x)
        
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x
