

# python evaluate.py --model_config=/workspace/BAFNet-plus/outputs/BAFNet_primek_0704/.hydra/config.yaml \
# --chkpt_dir=/workspace/BAFNet-plus/outputs/BAFNet_primek_0704/ \
# --chkpt_file=best.th \
# --noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_resampled  \
# --noise_test=dataset/noise_test.txt \
# --rir_dir=/workspace/dataset/datasets_fullband/impulse_responses \
# --rir_test=dataset/rir_test.txt \
# --test_augment_numb=2 \
# --snr_step -20 -10 0 10 15  \
# --eval_stt

python evaluate.py --model_config=/workspace/BAFNet-plus/outputs/PKNet_masking_0705/.hydra/config.yaml \
--chkpt_dir=/workspace/BAFNet-plus/outputs/PKNet_masking_0705 \
--chkpt_file=best.th \
--noise_dir=/workspace/dataset/datasets_fullband/noise_fullband_resampled  \
--noise_test=dataset/noise_test.txt \
--rir_dir=/workspace/dataset/datasets_fullband/impulse_responses \
--rir_test=dataset/rir_test.txt \
--test_augment_numb=2 \
--snr_step -10 0 10 15  \
--eval_stt