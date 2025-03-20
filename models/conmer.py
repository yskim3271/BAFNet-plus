import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Union, List

class Swish(nn.Module):
    """
    Swish 활성화 함수: x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    """
    Gated Linear Unit (GLU)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)

class ConvSubsampling(nn.Module):
    """
    CNN-based subsampling module (frontend)
    """
    def __init__(self, in_channels: int = 80, out_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(128 * (in_channels // 4), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T//4, out_channels]
        """
        # [B, T, D] -> [B, 1, T, D]
        x = x.unsqueeze(1)
        # [B, 1, T, D] -> [B, 128, T//2, D//2]
        x = self.relu1(self.conv1(x))
        # [B, 128, T//2, D//2] -> [B, 128, T//4, D//4]
        x = self.relu2(self.conv2(x))
        
        b, c, t, f = x.size()
        # [B, 128, T//4, D//4] -> [B, T//4, 128 * D//4]
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        # [B, T//4, 128 * D//4] -> [B, T//4, out_channels]
        x = self.linear(x)
        return x

class FeedForwardModule(nn.Module):
    """
    Feedforward module in Conformer block
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, hidden_dim)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.ff2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T, D]
        """
        residual = x
        x = self.layer_norm(x)
        x = self.ff1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.ff2(x)
        x = self.dropout2(x)
        return 0.5 * residual + x

class ConvModule(nn.Module):
    """
    Conformer convolution module
    """
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        
        # causal padding for Conv1d
        self.causal_padding = kernel_size - 1
        
        # pointwise convolution
        self.pointwise_conv1 = nn.Conv1d(
            dim, 2 * dim, kernel_size=1, stride=1, padding=0
        )
        self.glu = GLU(dim=1)
        
        # depthwise convolution (causal, no padding to the right)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, stride=1, 
            padding=0, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.swish = Swish()
        
        # pointwise convolution
        self.pointwise_conv2 = nn.Conv1d(
            dim, dim, kernel_size=1, stride=1, padding=0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T, D]
        """
        residual = x
        x = self.layer_norm(x)
        
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        # Apply pointwise conv
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Apply causal padding and then depthwise conv
        x = F.pad(x, (self.causal_padding, 0), 'constant', 0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        
        # Apply pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        return residual + x

class GatingConvModule(nn.Module):
    """
    Gating CNN module for Conmer-v1
    """
    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        
        # 첫 번째 Feedforward: dimension 축소 (dim -> hidden_dim)
        self.ff1 = nn.Linear(dim, hidden_dim)
        self.swish = Swish()
        
        # 두 번째 Feedforward
        self.ff2 = nn.Linear(hidden_dim, dim * 2)  # output dim을 2배로 하여 gating에 사용
        
        # causal padding for Conv1d
        self.causal_padding = kernel_size - 1
        
        # Depthwise CNN with GeLU
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, stride=1, 
            padding=0, groups=dim
        )
        self.gelu = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T, D]
        """
        residual = x
        x = self.layer_norm(x)
        
        # 첫 번째 Feedforward
        x = self.ff1(x)
        x = self.swish(x)
        
        # 두 번째 Feedforward
        x = self.ff2(x)
        
        # Split along channel dimension
        gate1, gate2 = x.chunk(2, dim=-1)
        
        # Apply depthwise conv to gate1
        # [B, T, D] -> [B, D, T]
        gate1 = gate1.transpose(1, 2)
        
        # Apply causal padding and then depthwise conv
        gate1 = F.pad(gate1, (self.causal_padding, 0), 'constant', 0)
        gate1 = self.depthwise_conv(gate1)
        gate1 = self.gelu(gate1)
        
        # [B, D, T] -> [B, T, D]
        gate1 = gate1.transpose(1, 2)
        
        # Element-wise multiplication (gating mechanism)
        x = gate1 * gate2
        
        x = self.dropout(x)
        
        return residual + x

class ConmerBlock(nn.Module):
    """
    Conmer 블록 (v0): MHSA를 제거한 Conformer 블록
    """
    def __init__(
        self, 
        dim: int, 
        ffn_dim: int, 
        conv_kernel_size: int = 31, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_dim, dropout)
        self.conv = ConvModule(dim, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T, D]
        """
        x = self.ffn1(x)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.layer_norm(x)
        return x

class ConmerBlockV1(nn.Module):
    """
    Conmer 블록 (v1): MHSA를 제거하고 gating CNN을 추가한 Conformer 블록
    """
    def __init__(
        self, 
        dim: int, 
        ffn_dim: int, 
        gating_hidden_dim: int = 256,
        conv_kernel_size: int = 32, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_dim, dropout)
        self.gating_conv = GatingConvModule(dim, gating_hidden_dim, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T, D]
        """
        x = self.ffn1(x)
        x = self.gating_conv(x)
        x = self.ffn2(x)
        x = self.layer_norm(x)
        return x

class ConmerEncoder(nn.Module):
    """
    Conmer 오디오 인코더 (v0, v1)
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_layers: int = 14,
        ffn_dim: int = 1024,
        gating_hidden_dim: int = 256,
        conv_kernel_size: int = 32,
        dropout: float = 0.1,
        version: str = 'v0'
    ):
        super().__init__()
        self.conv_subsampling = ConvSubsampling(input_dim, encoder_dim)
        
        if version == 'v0':
            self.layers = nn.ModuleList([
                ConmerBlock(
                    encoder_dim, ffn_dim, conv_kernel_size, dropout
                ) for _ in range(num_layers)
            ])
        elif version == 'v1':
            self.layers = nn.ModuleList([
                ConmerBlockV1(
                    encoder_dim, ffn_dim, gating_hidden_dim, conv_kernel_size, dropout
                ) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported Conmer version: {version}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, T, D]
        Returns:
            torch.Tensor: [B, T//4, encoder_dim]
        """
        x = self.conv_subsampling(x)
        for layer in self.layers:
            x = layer(x)
        return x

class LabelEncoder(nn.Module):
    """
    Unidirectional LSTM 기반 라벨 인코더
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 640, output_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y (torch.Tensor): [B, U]
        Returns:
            torch.Tensor: [B, U, output_dim]
        """
        y = self.embedding(y)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(y)
        y = self.linear(y)
        return y

class JointNetwork(nn.Module):
    """
    Transducer Joint Network
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim)
        self.joint_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, enc_out: torch.Tensor, dec_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_out (torch.Tensor): [B, T, encoder_dim]
            dec_out (torch.Tensor): [B, U, decoder_dim]
        Returns:
            torch.Tensor: [B, T, U, vocab_size]
        """
        # [B, T, encoder_dim] -> [B, T, hidden_dim]
        enc_out = self.encoder_proj(enc_out)
        # [B, U, decoder_dim] -> [B, U, hidden_dim]
        dec_out = self.decoder_proj(dec_out)
        
        # [B, T, hidden_dim] -> [B, T, 1, hidden_dim]
        enc_out = enc_out.unsqueeze(2)
        # [B, U, hidden_dim] -> [B, 1, U, hidden_dim]
        dec_out = dec_out.unsqueeze(1)
        
        # [B, T, 1, hidden_dim] + [B, 1, U, hidden_dim] -> [B, T, U, hidden_dim]
        joint_out = enc_out + dec_out
        joint_out = torch.tanh(joint_out)
        
        # [B, T, U, hidden_dim] -> [B, T, U, vocab_size]
        joint_out = self.joint_proj(joint_out)
        
        return joint_out

class ConmerTransducer(nn.Module):
    """
    Conmer Transducer (RNN-T)
    """
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_encoder_layers: int = 14,
        ffn_dim: int = 1024,
        gating_hidden_dim: int = 256,
        conv_kernel_size: int = 32,
        dropout: float = 0.1,
        vocab_size: int = 2500,
        label_hidden_dim: int = 640,
        joint_hidden_dim: int = 512,
        version: str = 'v0'
    ):
        super().__init__()
        
        # Audio Encoder
        self.encoder = ConmerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            ffn_dim=ffn_dim,
            gating_hidden_dim=gating_hidden_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            version=version
        )
        
        # Label Encoder
        self.decoder = LabelEncoder(
            vocab_size=vocab_size,
            hidden_dim=label_hidden_dim,
            output_dim=encoder_dim
        )
        
        # Joint Network
        self.joint = JointNetwork(
            encoder_dim=encoder_dim,
            decoder_dim=encoder_dim,
            hidden_dim=joint_hidden_dim,
            vocab_size=vocab_size
        )
    
    def forward(
        self, 
        speech: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            speech (torch.Tensor): [B, T, D]
            labels (torch.Tensor): [B, U]
        Returns:
            torch.Tensor: [B, T, U, vocab_size]
        """
        # Audio Encoder
        enc_out = self.encoder(speech)
        
        # Label Encoder
        dec_out = self.decoder(labels)
        
        # Joint Network
        joint_out = self.joint(enc_out, dec_out)
        
        return joint_out

    def compute_flops(self, seq_length=1000, label_length=100, input_dim=80):
        """
        Conmer 모델의 근사 FLOPs 계산
        """
        # 다운샘플링 이후 시퀀스 길이
        enc_seq_length = seq_length // 4
        
        # CNN Frontend FLOPs
        cnn_frontend_flops = (
            # Conv1
            seq_length * input_dim * 128 * 3 * 3 +
            # Conv2
            seq_length // 2 * input_dim // 2 * 128 * 128 * 3 * 3 +
            # Linear
            enc_seq_length * 128 * (input_dim // 4) * 256
        )
        
        # Encoder layer FLOPs
        flops_per_encoder_layer = 0
        
        # FFM1
        flops_per_encoder_layer += enc_seq_length * 256 * 1024 * 2  # 두 개의 Linear layer
        
        if self.encoder.version == 'v0':
            # Conv Module
            flops_per_encoder_layer += (
                # Pointwise conv1
                enc_seq_length * 256 * 512 +
                # Depthwise conv
                enc_seq_length * 256 * self.encoder.conv_kernel_size +
                # Pointwise conv2
                enc_seq_length * 256 * 256
            )
        else:  # v1
            # Gating Conv Module
            flops_per_encoder_layer += (
                # FF1
                enc_seq_length * 256 * self.encoder.gating_hidden_dim +
                # FF2
                enc_seq_length * self.encoder.gating_hidden_dim * 512 +
                # Depthwise conv
                enc_seq_length * 256 * self.encoder.conv_kernel_size
            )
        
        # FFM2
        flops_per_encoder_layer += enc_seq_length * 256 * 1024 * 2
        
        # Total encoder FLOPs
        encoder_flops = flops_per_encoder_layer * self.encoder.num_layers
        
        # Label encoder FLOPs
        decoder_flops = (
            # Embedding
            label_length * 640 +
            # LSTM
            label_length * 640 * 640 * 4 +
            # Linear
            label_length * 640 * 256
        )
        
        # Joint network FLOPs
        joint_flops = (
            # Encoder proj
            enc_seq_length * 256 * 512 +
            # Decoder proj
            label_length * 256 * 512 +
            # Joint proj
            enc_seq_length * label_length * 512 * 2500
        )
        
        total_flops = cnn_frontend_flops + encoder_flops + decoder_flops + joint_flops
        
        return {
            "cnn_frontend_flops": cnn_frontend_flops,
            "encoder_flops": encoder_flops,
            "decoder_flops": decoder_flops,
            "joint_flops": joint_flops,
            "total_flops": total_flops,
            "total_gflops": total_flops / 1e9
        }

# 테스트 함수
def test_conmer():
    # 테스트 데이터
    batch_size = 2
    time_length = 1000
    feature_dim = 80
    label_length = 100
    vocab_size = 2500
    
    # 음성 특징과 라벨 데이터 생성
    speech = torch.randn(batch_size, time_length, feature_dim)
    labels = torch.randint(0, vocab_size, (batch_size, label_length))
    
    print("=== Conmer-v0 테스트 ===")
    model_v0 = ConmerTransducer(
        input_dim=feature_dim,
        vocab_size=vocab_size,
        version='v0'
    )
    
    # 모델 출력 형태 확인
    with torch.no_grad():
        output_v0 = model_v0(speech, labels)
    
    print(f"입력 음성 형태: {speech.shape}")
    print(f"입력 라벨 형태: {labels.shape}")
    print(f"모델 출력 형태: {output_v0.shape}")
    
    # FLOPs 계산
    flops_v0 = model_v0.compute_flops(time_length, label_length, feature_dim)
    print(f"Conmer-v0 GFLOPs: {flops_v0['total_gflops']:.2f}")
    
    print("\n=== Conmer-v1 테스트 ===")
    model_v1 = ConmerTransducer(
        input_dim=feature_dim,
        vocab_size=vocab_size,
        version='v1'
    )
    
    # 모델 출력 형태 확인
    with torch.no_grad():
        output_v1 = model_v1(speech, labels)
    
    print(f"입력 음성 형태: {speech.shape}")
    print(f"입력 라벨 형태: {labels.shape}")
    print(f"모델 출력 형태: {output_v1.shape}")
    
    # FLOPs 계산
    flops_v1 = model_v1.compute_flops(time_length, label_length, feature_dim)
    print(f"Conmer-v1 GFLOPs: {flops_v1['total_gflops']:.2f}")
    
    print("\n모델 비교:")
    print(f"Conmer-v0 GFLOPs: {flops_v0['total_gflops']:.2f}")
    print(f"Conmer-v1 GFLOPs: {flops_v1['total_gflops']:.2f}")
    
    return model_v0, model_v1

if __name__ == "__main__":
    test_conmer()
