# import struct
# import sys
# import os
import json
# import string
# import sqlite3
# import collections
# import re
# import random
# from nltk.tokenize import sent_tokenize, word_tokenize
from glob import glob

"""
Convert JSON file to standalone files
"""

json_file_glob = "../../models/textsum/log-nips-abstract-title-decode/decode*"
json_files = glob(json_file_glob)
json_filename = json_files[-1] # use latest

summaries = [] # model-generated
references = [] # human-generated
articles = {}

print("Reading JSON file...\n")

with open(json_filename) as json_file:
  json_data = json_file.read()
  data = json.loads(json_data)
  print("%d entries" % len(data))
  for i, example in enumerate(data):
    datum = example['data']
    if not datum in articles:
      articles[datum] = True
      summaries.append(example['prediction'])
      references.append(example['label'])
    # else:
      # print("Duplicate at %d: %s" % (i, datum))
    # summaries.append(example['prediction'])
    # write "system" (predicted) file
    # system_filename = system_filename_template % i
    # with open(system_filename, "w") as system_file:
      # system_file.write(example['prediction'])
    # write "model" (ground truth) files
    # references.append(example['label'])
    # for label_i, label in enumerate(example['label']):
      # reference.append(label)
      # model = models[label_i]
      # model_filename = model_filename_template % (model, i)
      # with open(model_filename, "w") as model_file:
        # model_file.write(label)

print("%d entries were used" % len(summaries))
print("Computing ROUGE scores...")

"""
ROUGE scores
"""

from pythonrouge.pythonrouge import Pythonrouge

ROUGE_PATH = "/Users/ludwig/code/pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl"
ROUGE_DATA = "/Users/ludwig/code/pythonrouge/pythonrouge/RELEASE-1.5.5/data"

# initialize setting of ROUGE, eval ROUGE-1, 2, SU4, L
rouge = Pythonrouge(n_gram=2, ROUGE_SU4=False, ROUGE_L=True, stemming=False, stopwords=False, word_level=True, length_limit=False, length=50, use_cf=True, cf=95, scoring_formula="average", resampling=False, samples=500, favor=False, p=0.5)


# If you evaluate ROUGE by sentence list as above, set files=False
setting_file = rouge.setting(files=False, summary=summaries, reference=references)

# If you need only recall of ROUGE metrics, set recall_only=True
result = rouge.eval_rouge(setting_file, recall_only=False, ROUGE_path=ROUGE_PATH, data_path=ROUGE_DATA, f_measure_only=False)
print(result)