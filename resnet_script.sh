# Run train or test_fp depending on the argument passed to the script

if [ "$1" == "train" ]; then
    python train.py --ckp=tc_24 --encoder=resnet-ibn --config=config/resnet_ibn.yaml
elif [ "$1" == "test" ]; then
    python test_fp.py --query_lens=5,7,10,15,20 \
                  --text=tc27_clf_eval \
                  --test_dir=../datasets/sample_100/audio \
                  --encoder=resnet-ibn \
                  --config=config/resnet_ibn.yaml \
                  --map
else
    echo "Invalid argument"
fi