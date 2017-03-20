from os.path import normpath, basename, join, exists, expanduser
from os import makedirs
import gzip
import json
import re
from glob import glob

evaluation_filename = "prediction.json.gz"

ignored_paths = set(['../../data/glove/', '../../data/sentiment-treebank/']) # ignore e.g glove "dataset"
dataset_paths = list( set(glob("../../data/*/")) - ignored_paths )
dataset_names = [ basename(normpath((path))) for path in dataset_paths ]

print("Predicting on " + ", ".join(dataset_names))

for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
  dataset_file_path = join(dataset_path, 'data.json')
  if exists(dataset_file_path):
    print(dataset_name)
    with open(dataset_file_path) as dataset_file:
      json_data = dataset_file.read()
      data = json.loads(json_data)
      for example in data:
        parts = example['data'].split('</s>')
        example['prediction'] = parts[0] + " </s> </p> </d>"
        del example['data']
        del example['set']
    predictions_folder_path = "../../evaluation/first-sentence_" + dataset_name
    if not exists(predictions_folder_path):
      makedirs(predictions_folder_path)
    predictions_file_path = join(predictions_folder_path, evaluation_filename)
    with gzip.open(predictions_file_path, 'w') as predictions_file:
      predictions_file.write( json.dumps(data) )
