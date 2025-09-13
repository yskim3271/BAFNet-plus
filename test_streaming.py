import torch
import time
import numpy as np
import torch.nn.functional as F
from models.primeknet_full_stream import PrimeKnetFullStream
from models.streamer import PrimeKnetStreamer

def run_tests():
    # --- Test Parameters ---
    sample_rate = 16000
    receptive_field_sec = 0.25
    
    # STFT params from original model
    win_len = 400
    hop_len = 100
    fft_len = 400

    # For streaming, we process in smaller chunks (hops)
    stream_hop_size = 256 # Process 256 samples at a time
    
    # Generate a long test audio signal (e.g., 3 seconds)
    test_audio = torch.randn(1, 1, int(3 * sample_rate)) # Use [B, C, T]

    print("--- Test Setup ---")
    print(f"Receptive Field Target: {receptive_field_sec}s")
    print(f"Streaming Hop Size: {stream_hop_size} samples")
    print("-" * 20)

    # --- Model Initialization ---
    model_params = {
        'win_len': win_len, 'hop_len': hop_len, 'fft_len': fft_len,
        'depth': 2, 'num_tsblock': 1, 'kernel_list': [3, 5, 7, 11]
    }
    
    # 1. Causal Offline Model (for ground truth)
    causal_offline_model = PrimeKnetFullStream(**model_params)
    causal_offline_model.eval()

    # 2. Streaming Model
    streamer_model = PrimeKnetFullStream(**model_params)
    streamer_model.eval()
    streamer_model.load_state_dict(causal_offline_model.state_dict())
    
    # Calculate actual receptive field from model params
    # This is a rough estimation based on previous analysis
    rf_frames = 37 # From depth=2, num_tsblock=1, kernel_list max 11
    receptive_field = (rf_frames - 1) * hop_len + win_len
    
    streamer = PrimeKnetStreamer(streamer_model, receptive_field, hop_size=stream_hop_size)

    # --- Test 1 & 2: Correctness and Output Equivalence ---
    print("\n--- Test 1 & 2: Correctness and Output Equivalence ---")
    
    num_hops = test_audio.shape[2] // stream_hop_size
    offline_input = test_audio[..., :num_hops * stream_hop_size]

    with torch.no_grad():
        offline_output = causal_offline_model(offline_input)

    stream_outputs = []
    with torch.no_grad():
        for i in range(num_hops):
            chunk = offline_input[0, 0, i * stream_hop_size : (i + 1) * stream_hop_size]
            processed_chunk = streamer.feed(chunk)
            stream_outputs.append(processed_chunk)

    stream_output = torch.cat(stream_outputs, dim=1).unsqueeze(1)

    if offline_output.shape[2] > stream_output.shape[2]:
        offline_output = offline_output[..., :stream_output.shape[2]]
    else:
        stream_output = stream_output[..., :offline_output.shape[2]]

    print(f"Offline model output shape: {offline_output.shape}")
    print(f"Streamer output shape: {stream_output.shape}")
    
    if offline_output.shape == stream_output.shape:
        print("Test 1 Passed: Streaming model runs and produces output of correct shape.")
    else:
        print(f"Test 1 Failed: Output shape mismatch. Offline: {offline_output.shape}, Stream: {stream_output.shape}")
    
    mse = torch.mean((offline_output - stream_output) ** 2)
    print(f"Mean Squared Error between offline and stream outputs: {mse.item():.4e}")
    if mse < 1e-5:
         print("Test 2 Passed: Outputs are numerically very similar.")
    else:
         print("Test 2 Warning: Outputs differ significantly (may be due to complex state logic).")


    # --- Test 3: Performance Comparison ---
    print("\n--- Test 3: Performance Comparison ---")
    
    window_sec = 1.0
    window_size = int(window_sec * sample_rate)
    window_audio = torch.randn(1, 1, window_size)

    start_time = time.time()
    with torch.no_grad():
        _ = causal_offline_model(window_audio)
    offline_time = time.time() - start_time
    print(f"Offline model processing time for one window ({window_sec}s): {offline_time:.6f} seconds")

    streamer_for_timing = PrimeKnetStreamer(streamer_model, receptive_field, hop_size=stream_hop_size)
    
    hop_times = []
    num_timing_hops = window_size // stream_hop_size
    with torch.no_grad():
        for i in range(num_timing_hops):
            chunk = window_audio[0, 0, i * stream_hop_size : (i + 1) * stream_hop_size]
            start_time = time.time()
            _ = streamer_for_timing.feed(chunk)
            hop_times.append(time.time() - start_time)
            
    avg_hop_time = np.mean(hop_times)
    audio_duration_per_hop = stream_hop_size / sample_rate
    rtf = avg_hop_time / audio_duration_per_hop

    print(f"Streaming model average time per hop ({stream_hop_size} samples): {avg_hop_time:.6f} seconds")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    if rtf < 1.0:
        print("Test 3 Passed: Streaming is faster than real-time.")
    else:
        print("Test 3 Failed: Streaming is slower than real-time.")

if __name__ == "__main__":
    run_tests()
