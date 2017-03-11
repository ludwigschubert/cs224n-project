# cs224n-project
Final project for Winter 2017 CS224n class

# tensorflow/textsum

## running
```
python textsum/seq2seq_attention.py --mode=train --article_key=article --abstract_key=abstract --data_path=../data/nips-papers/titles_and_abstracts.pb2 --vocab_path=../data/nips-papers/vocab --log_root=textsum/log_root --train_dir=textsum/log_root/train
```