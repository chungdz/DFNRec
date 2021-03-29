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


python -m adressa_prepocess.new_data
python -m adressa_prepocess.build_dicts
python -m prepocess.find_neighbors --root=adressa
python -m prepocess.build_train --processes=10 --root=adressa --fsamples=train_behaviors.tsv --dataset=adressa
python -m prepocess.resplit --filenum 10 --processes 4 --fsamples=adressa/raw/train
python -m prepocess.build_valid --processes=10 --fsamples=dev_behaviors.tsv --root=adressa --dataset=adressa
python -m adressa_prepocess.resplit --filenum 10

CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --epoch=10 --filenum=1 --root=adressa
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --epoch=10 --filenum=10 --root=adressa

CUDA_VISIBLE_DEVICES=4,5,6,7 python deers_train.py --gpus=4 --epoch=10 --filenum=1 --root=adressa --port=9440
CUDA_VISIBLE_DEVICES=4,5,6,7 python deers_validate.py --gpus=4 --epoch=4 --filenum=10 --root=adressa