"""
Prepare data

Takes a dataset folder and converts/manipulates dataset as needed by model.
"""

# read arguments

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset_folder', metavar='dataset folder', type=str,
                    help='path to folder containing a data.json file')
args = parser.parse_args()

# read data

import json
import os
from collections import defaultdict, Counter
file_name = 'data.json'
file_path = os.path.join(args.dataset_folder, file_name)
# os.makedirs('data', exist_ok=True)

print("Loading dataset...")
with open(file_path,'r') as file_object:
  dataset = json.load(file_object)
  data = defaultdict(list)
  for entry in dataset:
    body = entry['data']
    for label in entry['label']:
      example = [body, label]
      data[entry['set']].append(example)

# write data files
from tensorflow.core.example import example_pb2
import struct

counter = Counter()

for key, values in data.items():
  print("%s:%d " % (key, len(values)))
  with open(os.path.join('data', key), 'wb') as writer:
    for body, label in values:
      # count words
      words = " ".join([body, label]).lower().split()
      counter.update(words)
      # encode as TF example
      tf_example = example_pb2.Example()
      assert(len(body) > 0)
      assert(len(label) > 0)
      tf_example.features.feature['article'].bytes_list.value.extend([str(body)])
      tf_example.features.feature['abstract'].bytes_list.value.extend([str(label)])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

with open(os.path.join('data', 'vocab'), 'w') as writer:
  # add 100000 most common words to vocab file
  for word, count in counter.most_common(100000):
    writer.write(word + ' ' + str(count) + '\n')
  # add special tokens required by textsum model
  for token in ['<UNK>', '<PAD>']:
    writer.write(token + ' 0\n')
