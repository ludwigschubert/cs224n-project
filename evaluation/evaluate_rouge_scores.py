from os.path import normpath, basename, join, exists, expanduser
import gzip
import json
from glob import glob
from termcolor import colored, cprint
from pythonrouge.pythonrouge import Pythonrouge

ROUGE = expanduser("~/code/pythonrouge/pythonrouge/RELEASE-1.5.5/")
ROUGE_PATH = join(ROUGE, "ROUGE-1.5.5.pl")
ROUGE_DATA = join(ROUGE, "data")

"""
Computes ROUGE scores for models and datasets that have outputs available
"""

evaluation_filename = "prediction.json.gz"
model_data_subfolder = "data"
model_training_subfolder = "train"

model_paths = glob("../models/*/")
model_names = [ basename(normpath((path))) for path in model_paths ]

glove_path = set(['../data/glove/']) # ignore glove "dataset"
dataset_paths = list( set(glob("../data/*/")) - glove_path )
dataset_names = [ basename(normpath((path))) for path in dataset_paths ]

def iterate_all_model_dataset_combos():
  evaluations = []
  for model_name, model_path in zip(model_names, model_paths):
    cprint("\nModel " + model_name, 'green', attrs=['bold'])
    cprint(40 * "-", 'green')
    for train_dataset_name in dataset_names:
      for test_dataset_name in dataset_names:
        evaluation_folder_name = "_".join([model_name, train_dataset_name, test_dataset_name])
        evaluation_file_name = join(evaluation_folder_name, evaluation_filename)
        if exists(evaluation_folder_name) and exists(evaluation_file_name):
          cprint("Evaluating trained on " + train_dataset_name + " tested on " + test_dataset_name, 'green')
          results = evaluate_rouge_scores(evaluation_file_name)
          results.update(model=model_name, dataset_trained=train_dataset_name, dataset_tested=test_dataset_name)
          evaluations.append(results)
        else:
          cprint("missing " + model_name + " trained on " + train_dataset_name + " tested on " + test_dataset_name, 'yellow')
  with open("evaluations.json", 'w') as evaluations_file:
    evaluations_file.write(json.dumps(evaluations))

def evaluate_rouge_scores(evaluation_file_name):
  summaries = [] # model-generated
  references = [] # human-generated
  # articles = {}
  with gzip.open(evaluation_file_name) as json_file:
    json_data = json_file.read()
    data = json.loads(json_data)
    print("%d entries..." % len(data))
    for example in data:
      # datum = example['data']
      # if not datum in articles:
        # articles[datum] = True
      summaries.append(example['prediction'].encode('utf-8').split())
      references.append([example.encode('utf-8').split() for example in example['label']])
  print("%d entries are used for evaluation." % len(summaries))
  # DEBUG: print a couple examples and their respective ROUGE scores
  # print(zip(summaries[5:10], references[5:10]))
  # rouge = Pythonrouge(n_gram=2, ROUGE_SU4=False, ROUGE_L=True, stemming=False, stopwords=False, word_level=True, length_limit=False, length=50, use_cf=True, cf=95, scoring_formula="average", resampling=False, samples=500, favor=False, p=0.5)
  # setting_file = rouge.setting(files=False, summary=summaries[5:10], reference=references[5:10])
  # print(rouge.eval_rouge(setting_file, recall_only=False, ROUGE_path=ROUGE_PATH, data_path=ROUGE_DATA, f_measure_only=False))
  rouge = Pythonrouge(n_gram=2, ROUGE_SU4=False, ROUGE_L=True, stemming=False, stopwords=False, word_level=True, length_limit=False, length=50, use_cf=True, cf=95, scoring_formula="average", resampling=False, samples=500, favor=False, p=0.5)
  setting_file = rouge.setting(files=False, summary=summaries, reference=references)
  result = rouge.eval_rouge(setting_file, recall_only=False, ROUGE_path=ROUGE_PATH, data_path=ROUGE_DATA, f_measure_only=False)
  return result


iterate_all_model_dataset_combos()