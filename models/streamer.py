import torch
from models.primeknet_stream import DummyPrimeKnetStream
from stft import mag_pha_stft, mag_pha_istft
import math

class PrimeKnetStreamer:
    def __init__(self, model, receptive_field, hop_size):
        self.model = model
        self.receptive_field = receptive_field
        self.hop_size = hop_size
        
        # Buffers and states
        self.input_buffer = torch.zeros(1, self.receptive_field)
        self.states = model.init_states()
        
        # STFT parameters from model
        self.win_len = model.win_len
        self.hop_len = model.hop_len
        self.fft_len = model.fft_len
        
        # Overlap-Add buffer
        # The overlap is win_len - hop_len
        self.output_buffer = torch.zeros(1, self.win_len - self.hop_len)
        self.win = torch.hann_window(self.win_len)

    def feed(self, audio_chunk):
        """
        Process a new audio chunk.
        Args:
            audio_chunk (Tensor): A tensor of shape [H] where H is hop_size.
        """
        if audio_chunk.shape[0] != self.hop_size:
            raise ValueError(f"Input chunk size must be {self.hop_size}")
            
        # 1. Update input buffer
        self.input_buffer = torch.roll(self.input_buffer, shifts=-self.hop_size, dims=1)
        self.input_buffer[0, -self.hop_size:] = audio_chunk

        # 2. Process the entire buffer to get a valid output for the new chunk
        return self._process_frame()

    def _process_frame(self):
        # STFT of the current input buffer
        # We process the whole receptive_field to get a valid output for the last hop
        mag, pha, _ = mag_pha_stft(
            self.input_buffer.squeeze(0), 
            n_fft=self.fft_len, 
            hop_size=self.hop_len, 
            win_size=self.win_len, 
            center=False
        )
        # mag from stft is already [B, F, T] where B=1, so no need to unsqueeze
        
        # Get the number of output frames from the last hop of audio
        num_new_frames = math.ceil(self.hop_size / self.hop_len)
        
        # We only need to process the new frames worth of spectrogram
        new_spec_chunk = mag[:, :, -num_new_frames:]

        # Streaming forward pass
        processed_spec_chunk, self.states = self.model.forward_stream(new_spec_chunk, self.states)
        
        # For this test, we will simplify and use the library iSTFT on the processed chunk.
        # A real implementation would require a proper Overlap-Add (OLA) mechanism for seamless stitching.
        output_wav = mag_pha_istft(
            processed_spec_chunk.squeeze(0), # Remove batch dim for istft function
            torch.zeros_like(processed_spec_chunk.squeeze(0)), 
            n_fft=self.fft_len, hop_size=self.hop_len, win_size=self.win_len
        )
        
        # We only care about the audio corresponding to the new frames
        return output_wav[..., -self.hop_size:]
