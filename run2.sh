# chmod +x ./run2.sh
# nohup ./run2.sh > output2.log 2>&1 &

echo Mapping Model
echo python train.py --config-name=config_mapping +model=primeknet_mapping dataset=TAPS loss.time=1.0
CUDA_VISIBLE_DEVICES=1 python train.py --config-name=config_mapping +model=primeknet_mapping dataset=TAPS loss.time=1.0