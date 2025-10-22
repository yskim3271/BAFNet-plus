# PrimeKnet 실시간 음성 향상 모델 성능 향상 계획

## 1. 프로젝트 목표

**핵심 목표**: 200ms latency budget 내에서 성능을 극대화하는 실시간 음성 향상 모델 개발

- **Target Latency**: 180-200ms
- **성능 최적화**: 200ms 여유를 최대한 활용하여 성능 향상
- **실시간 처리**: Causal 또는 near-streaming 처리 가능
- **기존 구조 활용**: PrimeKnet 기반 아키텍처 유지 및 개선

## 2. 현재 모델 분석

### 2.1 모델 구조 (`/home/yskim/workspace/BAFNet-plus/models/primeknet_lstm.py`)
- **Encoder-Decoder 구조**: Dense Encoder → Two Stage Blocks → Mask/Phase Decoder
- **Two Stage Block 구성**:
  - Time Stage: Channel Attention + LSTM_Group_Feature_Network
  - Freq Stage: Channel Attention + Group_Prime_Kernel_FFN
- **LSTM 설정**: 2 layers, hidden_size=64
- **특징**:
  - Magnitude masking + Phase prediction 방식
  - Causal mode 지원 (실시간 처리 가능)
  - Dense channel: 64
  - 4개의 Two Stage Block 사용

### 2.2 설정 파일
- FFT: 400, hop_size: 100, win_size: 400
- Sampling rate: 16kHz
- Loss weights: magnitude=0.9, phase=0.3, complex=0.1, consistency=0.05
- Batch size: 4, Learning rate: 0.0005
- Causal mode: true

### 2.3 실험 결과
- **PrimeKnet LSTM**: 기본 성능 확립
- **PrimeKnet Mamba**: 성능 하락 확인 → 제외
- **개선 필요성**: 200ms latency budget을 충분히 활용하지 못함

## 3. 최신 기술 동향 (2024-2025)

### 3.1 Lookahead Buffer Strategy
- **DarkStream (2024)**: 140-280ms lookahead buffer로 near-streaming 처리
- **효과**: Forward coarticulatory transitions 모델링으로 자연스러운 음성
- **장점**: 200ms budget 내에서 충분히 운용 가능
- **예상 향상**: Purely causal 대비 PESQ +0.15-0.2

### 3.2 Knowledge Distillation
- **RaD-Net 2 (2024)**: Non-causal teacher → Causal student 지식 전달
- **Causality-based KD**: Teacher의 future information을 student가 학습
- **검증된 성과**: OVRL DNSMOS +0.10 (ICASSP 2024 SSI Challenge)
- **핵심 아이디어**: Non-causal의 우수한 성능을 causal 모델로 전달

### 3.3 Advanced Attention Mechanisms
- **Complex Axial Self-Attention**: Frequency-wise long-range dependency 포착
- **Cross-Frame Attention**: Temporal context 모델링 강화
- **Causal Local Attention**: Efficiency와 receptive field 균형

## 4. 제안하는 개선 방안

### 4.1 Lookahead Buffer Strategy (최우선 구현)

#### **4.1.1 DarkStream 방식 Lookahead Layer**
```python
class LookaheadModule(nn.Module):
    def __init__(self, lookahead_ms=140, sr=16000, hop_size=100):
        """
        Args:
            lookahead_ms: Future context in milliseconds (140-280ms)
            sr: Sampling rate
            hop_size: STFT hop size
        """
        super().__init__()
        self.lookahead_frames = int(lookahead_ms / (hop_size / sr * 1000))

        # Causal encoder for past context
        self.causal_encoder = DenseEncoder()

        # Non-causal lookahead layer for future context
        self.lookahead_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=512
        )

    def forward(self, x, buffer):
        # Process past with causal encoder
        past_feat = self.causal_encoder(x)

        # Process with lookahead buffer
        with_future = torch.cat([x, buffer[:, :self.lookahead_frames]], dim=1)
        lookahead_feat = self.lookahead_layer(with_future)

        return lookahead_feat[:, :x.size(1)]  # Return only current frame
```
**예상 효과**:
- Latency: 140-180ms (200ms budget 내)
- PESQ: +0.15-0.2 (lookahead effect)
- 자연스러운 음성 재합성

### 4.2 Knowledge Distillation from Non-Causal Teacher

#### **4.2.1 Two-Stage Training Pipeline**
```python
class CausalityBasedKD:
    def __init__(self):
        # Stage 1: Train non-causal teacher (best performance)
        self.teacher = PrimeKnet(
            causal=False,
            bidirectional=True,
            num_tsblock=4,
            dense_channel=64
        )

        # Stage 2: Train causal student (with lookahead)
        self.student = PrimeKnet(
            causal=True,
            lookahead_ms=140,
            num_tsblock=4,
            dense_channel=64
        )

        # KD hyperparameters
        self.kd_loss_weight = 0.3
        self.temperature = 4.0

    def kd_loss(self, student_logits, teacher_logits, targets):
        # Soft target loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard target loss
        hard_loss = F.mse_loss(student_logits, targets)

        # Combined loss
        return (1 - self.kd_loss_weight) * hard_loss + \
               self.kd_loss_weight * soft_loss
```

#### **4.2.2 구현 단계**
1. **Non-causal Teacher 학습**:
   - Bidirectional LSTM/GRU 사용
   - Full context 활용하여 best performance 달성
   - VoiceBank+DEMAND에서 검증

2. **Causal Student Pre-training**:
   - Lookahead buffer와 함께 학습
   - Teacher 없이 초기 학습

3. **Knowledge Distillation**:
   - Teacher 모델 freeze
   - Student가 teacher의 behavior 모방
   - Feature-level + Output-level distillation

4. **Fine-tuning**:
   - Target dataset에서 fine-tuning
   - Distillation loss weight 조정

**예상 효과**:
- OVRL DNSMOS: +0.10-0.15 (RaD-Net 2 검증)
- Non-causal의 장점을 causal 모델로 전달
- 추가 latency 없이 성능 향상

### 4.3 Advanced Attention Mechanisms

#### **4.3.1 Complex Axial Self-Attention**
```python
class ComplexAxialAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        # Frequency-wise attention (non-causal)
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Time-wise causal local attention
        self.time_attention = CausalLocalAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            window_size=16
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, F, C]
        B, T, F, C = x.shape

        # Frequency-wise attention
        x_freq = x.permute(0, 1, 3, 2).reshape(B*T*C, F, -1)
        x_freq, _ = self.freq_attention(x_freq, x_freq, x_freq)
        x_freq = x_freq.reshape(B, T, C, F).permute(0, 1, 3, 2)
        x = self.layer_norm1(x + x_freq)

        # Time-wise causal attention
        x_time = x.permute(0, 2, 1, 3).reshape(B*F, T, C)
        x_time = self.time_attention(x_time)
        x_time = x_time.reshape(B, F, T, C).permute(0, 2, 1, 3)
        x = self.layer_norm2(x + x_time)

        return x
```

#### **4.3.2 Cross-Frame Attention with Buffer**
```python
class CrossFrameAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=8,
                 past_frames=8, future_frames=4):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Positional encoding for temporal information
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, x, past_buffer, future_buffer):
        # Concatenate past, current, and future frames
        context = torch.cat([
            past_buffer[:, -self.past_frames:],
            x,
            future_buffer[:, :self.future_frames]
        ], dim=1)

        # Add positional encoding
        context = self.pos_encoding(context)

        # Self-attention over full context
        attended, _ = self.attention(context, context, context)

        # Return only current frame
        start_idx = self.past_frames
        end_idx = start_idx + x.size(1)
        return attended[:, start_idx:end_idx]
```

**예상 효과**:
- Frequency-wise long-range dependency 포착
- Temporal context 강화
- SI-SDR: +1.5-2.0 dB

## 5. 참고 자료

### 5.1 핵심 논문

**Lookahead & Streaming**:
1. "DarkStream: real-time speech anonymization with low latency" (2024)
2. "Lookahead When It Matters: Adaptive Non-causal Transformers" (2023)
3. "Improving Streaming End-to-End ASR on Transformer-based Causal Models with Encoder States Revision Strategies" (2022)

**Knowledge Distillation**:
4. "RaD-Net 2: A causal two-stage repairing and denoising speech enhancement network with knowledge distillation and complex axial self-attention" (2024)
5. "Two-Step Knowledge Distillation for Tiny Speech Enhancement" (2023)

**Attention Mechanisms**:
6. "Supervised Attention Multi-Scale Temporal Convolutional Network for monaural speech enhancement" (2024)
7. "RaD-Net 2: Complex Axial Self-Attention for frequency-wise processing" (2024)

### 5.2 구현 참고

**Open-Source Repositories**:
- [RaD-Net 2 GitHub](https://github.com/) - Knowledge distillation reference
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Attention mechanisms
- [TorchAudio](https://pytorch.org/audio/) - STFT/iSTFT utilities

### 5.3 Dataset Resources
- **VoiceBank+DEMAND**: Standard benchmark
- **DNS Challenge**: Large-scale noisy speech
- **TAPS**: Current training dataset
- **LibriSpeech**: Clean speech for augmentation
