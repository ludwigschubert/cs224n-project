from os.path import normpath, basename, join, exists, expanduser
import gzip
import json
from glob import glob
from termcolor import colored, cprint
from pythonrouge.pythonrouge import Pythonrouge

ROUGE = expanduser("~/cs224n-project/evaluation/pythonrouge/pythonrouge/RELEASE-1.5.5/")
ROUGE_PATH = join(ROUGE, "ROUGE-1.5.5.pl")
ROUGE_DATA = join(ROUGE, "data")

"""
Computes ROUGE scores for models and datasets that have outputs available
"""

prediction_filename = "prediction.json.gz"
evaluation_filename = "evaluation.json"

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


evaluation_paths = glob("*/")
for evaluation_path in evaluation_paths:
  names = basename(normpath(evaluation_path)).split("_")
  if len(names) < 2:
    continue
  model_name, train_dataset_name = names[0], names[1]
  if len(names) > 2:
    test_dataset_name = names[2]
  else:
    test_dataset_name = train_dataset_name
  cprint("\nModel " + model_name, 'green', attrs=['bold'])
  cprint(40 * "-", 'green')
  prediction_file_path = join(evaluation_path, prediction_filename)
  evaluation_file_path = join(evaluation_path, evaluation_filename)
  if exists(prediction_file_path) and not exists(evaluation_file_path):
    cprint("Evaluating trained on " + train_dataset_name + " tested on " + test_dataset_name, 'green')
    results = evaluate_rouge_scores(prediction_file_path)
    print(results)
    results.update(model=model_name, dataset_trained=train_dataset_name, dataset_tested=test_dataset_name)
    with open(evaluation_file_path, 'w') as evaluations_file:
      evaluations_file.write(json.dumps(results))
