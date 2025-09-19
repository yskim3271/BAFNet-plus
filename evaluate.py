# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik kim
import torch
import nlptutti as sarmetric
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from compute_metrics import compute_metrics
from stft import mag_pha_stft, mag_pha_istft
from utils import bold, LogProgress


def get_stts(args, logger, enhanced):

    cer, wer = 0, 0
    model_id = "ghost613/whisper-large-v3-turbo-korean"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(args.device)
    
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float32,
        device=args.device,)
    
    iterator = LogProgress(logger, enhanced, name="STT Evaluation")
    for wav, text in iterator:
        with torch.no_grad():
            transcription = pipe(wav, generate_kwargs={"num_beams": 1, "max_length": 100})['text']

        cer += sarmetric.get_cer(text, transcription, rm_punctuation=True)['cer']
        wer += sarmetric.get_wer(text, transcription, rm_punctuation=True)['wer']
    
    cer /= len(enhanced)
    wer /= len(enhanced)
    
    return cer, wer

def evaluate(args, model, data_loader_list, logger, epoch=None, stft_args=None):
    

    prefix = f"Epoch {epoch+1}, " if epoch is not None else ""

    metrics = {}
    model.eval()
    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        enhanced = []
        results  = []
        with torch.no_grad():
            for data in iterator:
                bcs, noisy_acs, clean_acs, id, text = data
                
                if args.model.input_type == "acs":
                    input = mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)
                elif args.model.input_type == "bcs":
                    input = mag_pha_stft(bcs, **stft_args)[2].to(args.device)
                elif args.model.input_type == "acs+bcs":
                    input = mag_pha_stft(bcs, **stft_args)[2].to(args.device), mag_pha_stft(noisy_acs, **stft_args)[2].to(args.device)

                clean_mag_hat, clean_pha_hat, clean_com_hat = model(input)

                clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

                clean_acs = clean_acs.squeeze().detach().cpu().numpy()
                clean_hat = clean_hat.squeeze().detach().cpu().numpy()
                if len(clean_acs) != len(clean_hat):
                    length = min(len(clean_acs), len(clean_hat))
                    clean_acs = clean_acs[0:length]
                    clean_hat = clean_hat[0:length]

                enhanced.append((clean_hat, text[0]))
                results.append(compute_metrics(clean_acs, clean_hat))
        
        pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
        metrics[f'{snr}dB'] = {
            "pesq": pesq,
            "stoi": stoi,
            "csig": csig,
            "cbak": cbak,
            "covl": covl,
            "segSNR": segSNR
        }
        logger.info(bold(f"{prefix}Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))
                
        if args.eval_stt:
            cer, wer = get_stts(args, logger, enhanced)
            metrics[f'{snr}dB']['cer'] = cer
            metrics[f'{snr}dB']['wer'] = wer
            logger.info(bold(f"{prefix}Performance on {snr}dB: CER={cer:.4f}, WER={wer:.4f}"))
   
    return metrics



if __name__=="__main__":
    import os
    import logging
    import logging.config
    import argparse
    import importlib
    from data import TAPSnoisytdataset
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset, concatenate_datasets
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name. default is best.th")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the noise directory.")
    parser.add_argument("--noise_test", type=str, required=True, help="List of noise files for testing.")
    parser.add_argument("--rir_dir", type=str, required=True, help="Path to the RIR directory.")
    parser.add_argument("--rir_test", type=str, required=True, help="List of RIR files for testing.")
    parser.add_argument("--test_augment_numb", type=int, default=2, help="Number of test augmentations. default is 2")
    parser.add_argument("--snr_step", nargs="+", type=int, required=True, help="Signal to noise ratio. default is 0 dB")
    parser.add_argument("--reverb_proportion", type=float, default=0.0, help="Reverberation proportion. default is 0.5")
    parser.add_argument("--target_dB_FS", type=float, default=-25, help="Target dB FS. default is -25")
    parser.add_argument("--target_dB_FS_floating_value", type=float, default=0, help="Target dB FS floating value. default is 0")
    parser.add_argument("--silence_length", type=float, default=0.2, help="Silence length. default is 0.2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Specifies the device (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers. default is 5")
    parser.add_argument("--log_file", type=str, default="output.log", help="Log file name. default is output.log")
    parser.add_argument("--eval_stt", default=False, action="store_true", help="Evaluate STT performance")
    
    args = parser.parse_args()
    chkpt_dir = args.chkpt_dir
    chkpt_file = args.chkpt_file
    device = args.device

    log_file = args.log_file
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    conf = OmegaConf.load(args.model_config)
    conf.device = device
    conf.eval_stt = args.eval_stt
    
    model_args = conf.model
    model_lib = model_args.model_lib
    model_name = model_args.model_class
    module = importlib.import_module("models."+ model_lib)
    model_class = getattr(module, model_name)
    
    model = model_class(**model_args.param).to(device)
    chkpt = torch.load(os.path.join(chkpt_dir, chkpt_file), map_location=device)
    model.load_state_dict(chkpt['model'])
    tm_only = model_args.input_type == "tm"
    
    testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
    testset_list = [testset] * args.test_augment_numb
    testset = concatenate_datasets(testset_list)
    
    ev_loader_list = {}

    noise_test_list = [os.path.join(args.noise_dir, line.strip()) for line in open(args.noise_test, "r")]
    rir_test_list = [os.path.join(args.rir_dir, line.strip()) for line in open(args.rir_test, "r")]

    stft_args = {
        "n_fft": model_args.param.fft_len,
        "hop_size": model_args.param.hop_len,
        "win_size": model_args.param.win_len,
        "compress_factor": model_args.param.compress_factor
    }

    for fixed_snr in args.snr_step:
        ev_dataset = TAPSnoisytdataset(
            datapair_list= testset,
            noise_list= noise_test_list,
            rir_list= rir_test_list,
            snr_range= [fixed_snr, fixed_snr],
            reverb_proportion=args.reverb_proportion,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            tm_only=tm_only,
        )

        ev_loader = DataLoader(
            dataset=ev_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True
        )

        ev_loader_list[f"{fixed_snr}"] = ev_loader

    logger.info(f"Model: {model_name}")
    logger.info(f"Input type: {model_args.input_type}")
    logger.info(f"Checkpoint: {chkpt_dir}")
    logger.info(f"Device: {device}")
    
    evaluate(args=conf,
            model=model,
            data_loader_list=ev_loader_list,
            logger=logger,
            epoch=None,
            stft_args=stft_args)
