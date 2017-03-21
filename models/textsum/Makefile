# ../../data/nips-abstract-title/data.json: ../../data/nips-abstract-title/convert.py
#     python3 ../../data/nips-abstract-title/convert.py

data/vocab data/train data/dev data/test: prepare_data.py ../../data/duc2004/data.json # also generates other required data files
	python prepare_data.py --dataset=../../data/duc2004

.PHONY: olddata
olddata:
	python data_convert_example.py \
	  --db ../../data/sources/nips-papers/database.sqlite \
	  --out_file data/titles_and_abstracts.pb2 \
	  --vocab_file data/vocab-old

.PHONY: train
train: data/vocab data/train
	python seq2seq_attention.py \
	  --mode=train \
	  --article_key=article \
	  --abstract_key=abstract \
	  --data_path=data/train \
	  --vocab_path=data/vocab \
	  --log_root=log \
	  --train_dir=log/train

.PHONY: eval
eval: data/vocab data/dev
	python seq2seq_attention.py \
	  --mode=eval \
	  --article_key=article \
	  --abstract_key=abstract \
	  --data_path=data/train \
	  --vocab_path=data/vocab \
	  --log_root=log-nips-abstract-title \
	  --train_dir=log-nips-abstract-title

 .PHONY: test
 test:
	python seq2seq_attention.py \
	  --mode=decode \
	  --article_key=article \
	  --abstract_key=abstract \
	  --data_path=data-nips/dev \
	  --vocab_path=data-nips/vocab \
	  --log_root=log-squad \
	  --decode_dir=../../evaluation/textsum_squad_nips \
	  --beam_size=2

.PHONY: clean
clean:
	rm -r data/*
	rm -r log/*