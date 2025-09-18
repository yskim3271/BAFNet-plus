# chmod +x ./run2.sh
# nohup ./run2.sh > output1.log 2>&1 &

echo Masking Model Causal
echo train.py +model=primeknet_masking model.param.causal=True model.param.dense_depth=3 continue_from=/home/work/yskim/BAFNet-plus/outputs/2025-09-15_18-47-09
python train.py +model=primeknet_masking model.param.causal=True model.param.dense_depth=3 continue_from=/home/work/yskim/BAFNet-plus/outputs/2025-09-15_18-47-09