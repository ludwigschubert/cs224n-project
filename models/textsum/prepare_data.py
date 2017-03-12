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
from collections import defaultdict
file_name = 'data.json'
file_path = os.path.join(args.dataset_folder, file_name)

print("Loading dataset…")
with open(file_path,'r') as file_object:
  dataset = json.load(file_object)
  data = defaultdict(list)
  for entry in dataset:
    body = entry['data']
    for label in entry['label']:
      example = { 'data': body, 'label': label }
      data[entry['set']].append(example)
  for key, values in data.items():
    print("%s:%d " % (key, len(values)))

# write data files
