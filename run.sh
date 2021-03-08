python -m prepocess.cut_train
python -m prepocess.build_dicts
python -m prepocess.find_neighbors
python -m prepocess.build_train --processes=10
python -m prepocess.resplit --filenum 10 --processes 4
python -m prepocess.build_valid --processes=10 --fsamples=dev/target_behaviors.tsv
CUDA_VISIBLE_DEVICES=4,5,6,7 python training.py --gpus=4 --epoch=10 --filenum=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --epoch=10 --filenum=10

CUDA_VISIBLE_DEVICES=4,5,6,7 python deers_train.py --gpus=4 --epoch=10 --filenum=1
CUDA_VISIBLE_DEVICES=4,5,6,7 python deers_validate.py --gpus=4 --epoch=4 --filenum=10