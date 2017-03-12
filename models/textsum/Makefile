../../data/nips-abstract-title/data.json: ../../data/nips-abstract-title/convert.py
	python3 ../../data/nips-abstract-title/convert.py

data/vocab data/train data/dev data/test: prepare_data.py ../../data/nips-abstract-title/data.json # also generates other required data files
	python prepare_data.py --dataset=../../data/nips-abstract-title

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
	  --data_path=data/titles_and_abstracts.pb2 \
	  --vocab_path=data/vocab \
	  --log_root=log \
	  --train_dir=log/train

.PHONY: eval
eval: data/vocab data/dev
	python seq2seq_attention.py \
	  --mode=eval \
	  --article_key=article \
	  --abstract_key=abstract \
	  --data_path=data/dev \
	  --vocab_path=data/vocab \
	  --log_root=log \
	  --train_dir=log/eval

 .PHONY: test
 test: data/vocab data/test
	python seq2seq_attention.py \
	  --mode=decode \
	  --article_key=article \
	  --abstract_key=abstract \
	  --data_path=data/test \
	  --vocab_path=data/vocab \
	  --log_root=log \
	  --train_dir=log/test

.PHONY: clean
clean:
	rm -r data/*
	rm -r log/*