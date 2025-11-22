CUDA_VISIBLE_DEVICES=1 python src/evaluate.py --model_config /home/yskim/workspace/BAFNet-plus/results/experiments/prk_taps_mask/.hydra/config.yaml \
--chkpt_dir /home/yskim/workspace/BAFNet-plus/results/experiments/prk_taps_mask \
--chkpt_file best.th \
--noise_dir /home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k \
--noise_test /home/yskim/workspace/BAFNet-plus/dataset/taps/noise_test.txt \
--rir_dir /home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k \
--rir_test /home/yskim/workspace/BAFNet-plus/dataset/taps/rir_test.txt \
--snr_step 0 \
--device cuda \
--eval_stt 