import pandas as pd

import pytreebank
import json
#dataset = pytreebank.load_sst()
input_files = ['trees/train.txt','trees/dev.txt','trees/test.txt']
dataset = {key.split('/')[-1].split('.')[0]: pytreebank.import_tree_corpus(key) for key in input_files}
dataset = {key: [(sorted([(len(s),s) for s in t1.to_lines()])[::-1][0][1],t1.label)
 for t1 in dataset[key]] for key in dataset}

dout = []
for key in dataset:
	dout += [{'data':x[0],'label':[x[1]],'set':key} for x in dataset[key]]
with open('data.json','w') as fp:
	json.dump(dout, fp,sort_keys=True,indent=4, separators=(',', ': '))
#sentences = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt',sep='\t')
#splits = pd.read_csv('stanfordSentimentTreebank/datasetSplit.txt',sep=',')
#labels = pd.read_csv('stanfordSentimentTreebank/sentiment_labels.txt',sep='|')
#phrases = pd.read_csv('stanfordSentimentTreebank/dictionary.txt',sep='|')