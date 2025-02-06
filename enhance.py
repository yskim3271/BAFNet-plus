

import os
import torch
import torchaudio

from matplotlib import pyplot as plt
from utils import LogProgress, mel_spectrogram

def save_wavs(wavs_dict, filepath, sr=16_000):
    for i, (key, wav) in enumerate(wavs_dict.items()):
        torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        
def save_mels(wavs_dict, filepath):
    num_mels = len(wavs_dict)
    figure, axes = plt.subplots(num_mels, 1, figsize=(10, 10))
    figure.set_tight_layout(True)
    figure.suptitle(filepath)
    
    for i, (key, wav) in enumerate(wavs_dict.items()):
        mel = mel_spectrogram(wav, device='cpu', sampling_rate=16_000)
        axes[i].imshow(mel.squeeze().numpy(), aspect='auto', origin='lower')
        axes[i].set_title(key)
    
    figure.savefig(filepath)
    plt.close(figure)

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def enhance(args, model, data_loader_list, epoch, logger, local_out_dir= None):
    model.eval()
    
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
    
    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Enhance on {snr}dB")
        outdir_mels= os.path.join(out_dir, f"mels_epoch{epoch+1}_{snr}dB")
        outdir_wavs= os.path.join(out_dir, f"wavs_epoch{epoch+1}_{snr}dB")
        os.makedirs(outdir_mels, exist_ok=True)
        os.makedirs(outdir_wavs, exist_ok=True)
        
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
            for data in iterator:
                # Get batch data (batch, channel, time)
                tm, noisy_am, clean_am, id, text = data
                            
                if args.model.input_type == "am":
                    clean_am_hat = model(noisy_am.to(args.device))
                elif args.model.input_type == "tm":
                    clean_am_hat = model(tm.to(args.device))
                elif args.model.input_type == "am+tm":
                    clean_am_hat = model(tm.to(args.device), noisy_am.to(args.device))
                else:
                    raise ValueError("Invalid model input type argument")
                
                tm = tm.squeeze().cpu()
                clean_am = clean_am.squeeze().cpu()
                noisy_am = noisy_am.squeeze().cpu()
                clean_am_hat = clean_am_hat.squeeze().cpu()
                            
                wavs_dict = {
                    "tm": tm,
                    "noisy_am": noisy_am,
                    "clean_am": clean_am,
                    "clean_am_hat": clean_am_hat,
                }
                
                save_wavs(wavs_dict, os.path.join(outdir_wavs, id))
                save_mels(wavs_dict, os.path.join(outdir_mels, id))
            
            
            
        