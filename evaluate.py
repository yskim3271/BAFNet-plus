# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik
import io
import os
import logging
import torch
import requests
import nlptutti as sarmetric
import numpy as np

from scipy.io.wavfile import write
from concurrent.futures import ThreadPoolExecutor
from pesq import pesq
from pystoi import stoi
from metric_helper import wss, llr, SSNR, trim_mos
from utils import bold, LogProgress

def _numpy_to_wavobject(ndarr, sample_rate=16000):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    assert ndarr.ndim == 1
    ndarr = ndarr / max(ndarr.max(), 1)
    ndarr = ndarr * 32767
    ndarr = ndarr.astype('int16')
    
    write(byte_io, sample_rate, ndarr)
    result_bytes = byte_io.read()
    return result_bytes

def request_transcript(data, url, headers):
    response = requests.post(url,  data=data, headers=headers)
    rescode = response.status_code
    if(rescode == 200):
        # clova stt api returns json format, so we need to remove the first 9 characters and the last 2 characters
        return response.text[9:-2]     
    else:
        logging.error("Error : " + response.text)
        return None

def parse_transcripts(filepath):
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
    return {l.split('|')[0]: l.split('|')[1] for l in lines}

def get_stts(args, data):
    transcripts = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {}
        for filename, d in data.items():
            d = _numpy_to_wavobject(d)
            future = executor.submit(request_transcript, d, args.url, args.headers)
            futures[filename] = future
        for filename, future in futures.items():
            transcript = future.result()
            transcripts[filename] = transcript
    return transcripts

## Code modified from https://github.com/wooseok-shin/MetricGAN-plus-pytorch/tree/main
def compute_metrics(target_wav, pred_wav, fs=16000):
    
    Stoi = stoi(target_wav, pred_wav, fs, extended=False)
    Pesq = pesq(ref=target_wav, deg=pred_wav, fs=fs)
        
    alpha = 0.95
    ## Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])
    
    ## Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])
    
    ## Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, 16000)
    segSNR = np.mean(segsnr_mean)
    
    ## Csig
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * Pesq - 0.009 * wss_dist
    Csig = float(trim_mos(Csig))
    
    ## Cbak
    Cbak = 1.634 + 0.478 * Pesq - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)

    ## Covl
    Covl = 1.594 + 0.805 * Pesq - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    
    return Pesq, Stoi, Csig, Cbak, Covl



def evaluate(args, model, data_loader, epoch, logger, local_out_dir=None):
        
    metrics = {}
    enhanced = {}
    model.eval()
    result = []
    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Evaluate enhanced files")
        for i, data in enumerate(iterator):
            # Get batch data
            x, noisy_y, clean_y, _, fileidx = data
            x = x.to(args.device)
            noisy_y = noisy_y.to(args.device)
            
            clean_y_hat = model(x, noisy_y)
            clean_y_hat = clean_y_hat.unsqueeze(1)
                        
            clean_y, clean_y_hat = clean_y.squeeze().cpu().numpy(), clean_y_hat.squeeze().cpu().numpy()
            
            enhanced[fileidx[0]] = clean_y_hat
            result.append(compute_metrics(clean_y, clean_y_hat))
            
    results = np.array(result)
    pesq, stoi, csig, cbak, covl = np.mean(results, axis=0)
    metrics = {'pesq': pesq, 'stoi': stoi, 'csig': csig, 'cbak': cbak, 'covl': covl}
    
    logger.info(bold(f'Test set performance:PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}'))
    
    if local_out_dir:
        out_dir = local_out_dir
        os.makedirs(out_dir, exist_ok=True)
    
    if args.eval_stt:
        cer = 0
        wer = 0
        transcripts_ref = parse_transcripts(args.transcripts)
        transcripts_gen = get_stts(args, enhanced)
        if local_out_dir:
            transcripts_file = os.path.join(out_dir, f'transcripts_{epoch}.txt')
        else:
            transcripts_file = f'transcripts_{epoch}.txt'
        with open(transcripts_file, 'w') as f:
            for fileidx, transcript in transcripts_gen.items():
                f.write(f'{fileidx}|{transcript}\n')
        for fileidx, transcript in transcripts_gen.items():
            if transcript is None:
                continue
            ref = transcripts_ref[fileidx]
            cer += sarmetric.get_cer(ref, transcript)['cer']
            wer += sarmetric.get_wer(ref, transcript)['wer']
        cer = cer / len(transcripts_gen)
        wer = wer / len(transcripts_gen)
        logger.info(bold(f'Test set performance:CER={cer:.4f}, WER={wer:.4f}.'))
        metrics['cer'] = cer
        metrics['wer'] = wer
   
    return metrics


