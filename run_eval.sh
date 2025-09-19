CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_config /home/yskim/workspace/BAFNet-plus/outputs/2025-09-14_21-06-34/.hydra/config.yaml \
--chkpt_dir /home/yskim/workspace/BAFNet-plus/outputs/2025-09-14_21-06-34 \
--chkpt_file best.th \
--noise_dir /home/yskim/workspace/dataset/datasets_fullband/noise_fullband_16k \
--noise_test /home/yskim/workspace/BAFNet-plus/dataset/noise_test.txt \
--rir_dir /home/yskim/workspace/dataset/datasets_fullband/impulse_responses_16k \
--rir_test /home/yskim/workspace/BAFNet-plus/dataset/rir_test.txt \
--snr_step=0 \
--device cuda \
--eval_stt