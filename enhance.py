

import os
import torch
import torchaudio

from matplotlib import pyplot as plt
from utils import LogProgress, mel_spectrogram


def save_wavs(noisy_y, clean_y, clean_y_hat, fileidx, out_dir, epoch, sampling_rate=16000):
    filename = os.path.join(out_dir, fileidx + f"_epoch{epoch+1}")
    write(noisy_y, filename + "_noisy.wav", sr=sampling_rate)
    write(clean_y, filename + "_clean.wav", sr=sampling_rate)
    write(clean_y_hat, filename + "_clean_hat.wav", sr=sampling_rate)

def save_mels(noisy_y_mel, clean_y_mel, clean_y_hat_mel, fileidx, out_dir, epoch):
    filename = os.path.join(out_dir, fileidx + f"_epoch{epoch}")
    figure, axes = plt.subplots(3, 1, figsize=(10, 10))
    figure.set_tight_layout(True)
    figure.suptitle(fileidx)
    
    axes[0].imshow(noisy_y_mel.squeeze().numpy(), aspect='auto', origin='lower')
    axes[1].imshow(clean_y_mel.squeeze().numpy(), aspect='auto', origin='lower')
    axes[2].imshow(clean_y_hat_mel.squeeze().numpy(), aspect='auto', origin='lower')
    
    axes[0].set_title('noisy')
    axes[1].set_title('clean')
    axes[2].set_title('clean_hat')

    figure.savefig(filename + "_mel.png")
    plt.close(figure)

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def enhance(args, model, data_loader, epoch, logger, local_out_dir= None):
    model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
        
    outdir_mels= os.path.join(out_dir, f"mels_epoch{epoch+1}")
    outdir_wavs= os.path.join(out_dir, f"wavs_epoch{epoch+1}")
    os.makedirs(outdir_mels, exist_ok=True)
    os.makedirs(outdir_wavs, exist_ok=True)

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        for data in iterator:
            # Get batch data (batch, channel, time)
            x, noisy_y, clean_y, _, fileidx = data
                        
            fileidx = fileidx[0]
            x = x.to(args.device)
            noisy_y = noisy_y.to(args.device)
            clean_y = clean_y.to(args.device)

            clean_y_hat = model(x, noisy_y)
            
            noisy_y = noisy_y.squeeze(1).cpu()
            clean_y = clean_y.squeeze(1).cpu()
            clean_y_hat = clean_y_hat.squeeze(1).cpu()
                        
            noisy_y_mel = mel_spectrogram(noisy_y, device='cpu', sampling_rate=args.sampling_rate)
            clean_y_mel = mel_spectrogram(clean_y, device='cpu', sampling_rate=args.sampling_rate)
            clean_y_hat_mel = mel_spectrogram(clean_y_hat, device='cpu', sampling_rate=args.sampling_rate)

            save_wavs(noisy_y, clean_y, clean_y_hat, fileidx, outdir_wavs, epoch, sampling_rate=args.sampling_rate)
            save_mels(noisy_y_mel, clean_y_mel, clean_y_hat_mel, fileidx, outdir_mels, epoch)
            
            
            
            
        