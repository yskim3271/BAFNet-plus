CUDA_VISIBLE_DEVICES=0 python enhance.py \
--dataset taps \
--chkpt_dir /home/yskim/workspace/BAFNet-plus/outputs/prk_taps_map \
--chkpt_file best.th \
--noise_dir /home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k \
--noise_test /home/yskim/workspace/BAFNet-plus/dataset/taps/noise_test.txt \
--rir_dir /home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k \
--rir_test /home/yskim/workspace/BAFNet-plus/dataset/taps/rir_test.txt \
--snr 0 \
--reverb_prop 0 \
--target_dB_FS -25 \
--output_dir prk_taps_map \
--device cuda

CUDA_VISIBLE_DEVICES=0 python enhance.py \
--dataset vibravox \
--chkpt_dir /home/yskim/workspace/BAFNet-plus/outputs/prk_vibravox_map \
--chkpt_file best.th \
--noise_dir /home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k \
--noise_test /home/yskim/workspace/BAFNet-plus/dataset/vibravox/noise_test.txt \
--rir_dir /home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k \
--rir_test /home/yskim/workspace/BAFNet-plus/dataset/vibravox/rir_test.txt \
--snr 0 \
--reverb_prop 0 \
--target_dB_FS -25 \
--output_dir prk_vibravox_map \
--device cuda
