# chmod +x ./run2.sh
# nohup ./run2.sh > output2.log 2>&1 &


RUNPOD_POD_ID=rg3atrs8rmnlum

echo python train.py --config-name=config_mapping +model=primeknet_mapping dataset=Vibravox eval_stt=False
python train.py --config-name=config_mapping +model=primeknet_mapping dataset=Vibravox eval_stt=False