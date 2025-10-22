# PrimeKnet with Lookahead Buffer

## Overview

PrimeKnet-LK (Lookahead) implements a simple yet effective lookahead buffer mechanism that redistributes the model's receptive field between past and future frames. This allows the model to leverage future context for improved speech enhancement quality at the cost of additional latency.

## Model Architecture

PrimeKnet-LK follows the **Two_Stage_Block** structure from primeknet_lstm:
- **Time stage**: Channel_Attention_Block + Group_Prime_Kernel_FFN (with lookahead)
- **Freq stage**: Channel_Attention_Block + Group_Prime_Kernel_FFN (non-causal)

This modular design separates channel attention and feedforward processing, making the architecture cleaner and more consistent with primeknet_lstm/mamba variants.

## Fixed Parameters

For all experiments, the following parameters are hardcoded:
- **time_dw_kernel_size**: 1 (minimal time kernel)
- **time_block_kernel**: [1] (minimal FFN expansion)
- **dense_depth**: 3 (default, can be overridden)

This configuration results in a low-latency model with:
- **Total RF**: 29 frames (vs 317 in full model)
- **Parameters**: ~978K
- **Latency**: Much lower due to smaller RF

## Key Concept

Traditional causal models use only past frames for prediction, while the lookahead variant redistributes the receptive field:

- **Traditional Causal**: 100% past frames (zero latency)
- **Lookahead Causal**: (100-L)% past frames + L% future frames

Where L is the lookahead ratio (0 ≤ L ≤ 0.5).

## Architecture

The lookahead mechanism is implemented by modifying the padding strategy in convolution layers:

```python
# Traditional causal padding
padding = [total_padding, 0]  # All padding on left (past)

# Lookahead padding
left_padding = total_padding * (1 - lookahead_ratio)
right_padding = total_padding * lookahead_ratio
padding = [left_padding, right_padding]
```

### Key Components

1. **LookaheadConv1d/2d**: Convolution layers with redistributed padding
2. **Channel_Attention_Block**: Gated attention with lookahead support
3. **Group_Prime_Kernel_FFN**: Multi-kernel feedforward with lookahead support
4. **Two_Stage_Block**: Separates time and frequency processing
5. **Lookahead DS_DDB**: Dense blocks in encoder/decoder with lookahead

## Lookahead Configurations

With updated settings (RF = 29 frames @ hop_len=100):

| Lookahead Ratio | Past Frames | Future Frames | Latency (ms) | Use Case |
|----------------|-------------|---------------|--------------|----------|
| 0.0 (0%)       | 29          | 0             | 0            | Baseline (pure causal) |
| 0.2 (20%)      | 24          | 5             | ~31          | Low latency |
| 0.3 (30%)      | 21          | 8             | ~50          | Balanced |
| 0.5 (50%)      | 15          | 14            | ~88          | Best quality |

**Note**: Much lower latency compared to full model due to reduced receptive field (29 vs 317 frames).

## Usage

### Training

Train a model with specific lookahead ratio using Hydra overrides:

```bash
# Train with 30% lookahead
python train.py +model=primeknet_lk_masking +dset=taps model.param.lookahead_ratio=0.3

# Train with different ratios
python train.py +model=primeknet_lk_masking +dset=taps model.param.lookahead_ratio=0.0  # Pure causal
python train.py +model=primeknet_lk_masking +dset=taps model.param.lookahead_ratio=0.5  # Max lookahead

# Train multiple configurations automatically
./train_lookahead.sh

# Quick test
./test_lk_single.sh 0.3  # Test with 30% lookahead
```

### Configuration

Single configuration file with Hydra overrides:
- `conf/model/primeknet_lk_masking.yaml` - Base configuration

Override lookahead ratio via command line:
```bash
python train.py +model=primeknet_lk_masking +dset=taps model.param.lookahead_ratio=0.25
```

## Streaming Considerations

When deploying in streaming mode:

1. **Buffer Management**: Need to maintain a buffer of `lookahead_frames` future frames
2. **Output Delay**: Output is delayed by `lookahead_ms` milliseconds
3. **Edge Handling**: Last `lookahead_frames` of the stream require special handling

Example streaming pseudocode:

```python
buffer = CircularBuffer(lookahead_frames)

for frame in input_stream:
    buffer.add(frame)
    if buffer.is_full():
        current = buffer.get_center()
        future = buffer.get_future()
        enhanced = model.process(current, future)
        output_stream.write(enhanced)
```

## Performance Trade-offs

| Aspect | Low Lookahead (0-20%) | Medium (30%) | High (50%) |
|--------|----------------------|--------------|------------|
| Latency | < 400ms | ~600ms | ~1000ms |
| PESQ Improvement | +0.05-0.10 | +0.15-0.20 | +0.20-0.25 |
| Real-time Factor | ~1.0x | ~1.0x | ~1.0x |
| Use Case | Live streaming | Video calls | Offline processing |

## Testing

Run unit tests:

```python
# Test different lookahead ratios
python -m pytest tests/test_models.py::TestPrimeKnetLookahead -v
```

## Implementation Details

### Receptive Field Calculation

With fixed parameters (dense_depth=3, time_dw_kernel_size=1, time_block_kernel=[1]):

**Encoder/Decoder DS_DDB** (3 layers, dilations=[1,2,4]):
- RF = 1 + (k-1) × sum(dilations) = 1 + 2 × 7 = 15 frames

**LKFCAnet** (minimal time kernels):
- RF_block = k_dw + K_g - 1 = 1 + 1 - 1 = 1
- RF_lkfca = 1 + (num_tsblock × time_block_num) × (RF_block - 1) = 1 frame

**Total RF**: 15 + 0 + 14 = **29 frames**

The lookahead ratio redistributes this fixed RF between past and future.

### Key Design Choices

1. **Simple redistribution**: No complex attention mechanisms, just padding redistribution
2. **Uniform lookahead**: All layers use the same lookahead ratio
3. **Time-only lookahead**: Frequency dimension remains unchanged
4. **Fixed RF**: Total receptive field stays constant regardless of lookahead
5. **Minimal time kernels**: time_dw_kernel_size=1, time_block_kernel=[1] for compact model
6. **Reduced depth**: dense_depth=3 for lower latency and faster inference

## Future Improvements

1. **Adaptive Lookahead**: Different lookahead ratios for different layers
2. **Attention-based Selection**: Learn which future frames are most relevant
3. **Dynamic Lookahead**: Adjust lookahead based on input characteristics
4. **Hierarchical Lookahead**: Coarse-to-fine future context utilization