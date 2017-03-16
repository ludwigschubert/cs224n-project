.PHONY: train train1 train2 train3 train21 train22 train23 train31 train32 train33 train41 train42 train43
train:
	train1
	train2
	train3
	train21
	train22
	train23
	train31
	train32
	train33
	train41
	train42
	train43

train-1:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=duc2004 \
	  --cnn=0 \
	  --train_embedding=0 \
	  --output_root="." \

train-2:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=nips-abstract-title \
	  --cnn=0 \
	  --train_embedding=0 \
	  --output_root="." \

train-3:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=squad \
	  --cnn=0 \
	  --train_embedding=0 \
	  --output_root="." \


train-21:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=duc2004 \
	  --cnn=1 \
	  --train_embedding=0 \
	  --output_root="." \

train-22:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=nips-abstract-title \
	  --cnn=1 \
	  --train_embedding=0 \
	  --output_root="." \

train-23:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=squad \
	  --cnn=1 \
	  --train_embedding=0 \
	  --output_root="." \



train-31:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=duc2004 \
	  --cnn=1 \
	  --train_embedding=1 \
	  --output_root="." \

train-32:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=nips-abstract-title \
	  --cnn=1 \
	  --train_embedding=1 \
	  --output_root="." \

train-33:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=squad \
	  --cnn=1 \
	  --train_embedding=1 \
	  --output_root="." \




train-41:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=duc2004 \
	  --cnn=0 \
	  --train_embedding=1 \
	  --output_root="." \

train-42:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=nips-abstract-title \
	  --cnn=0 \
	  --train_embedding=1 \
	  --output_root="." \

train-43:
	python decoderTT.py \
	  --runmode=train \
	  --dataset_name=squad \
	  --cnn=0 \
	  --train_embedding=1 \
	  --output_root="." \