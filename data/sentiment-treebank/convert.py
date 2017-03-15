import pandas as pd

import pytreebank
import json
import json
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize_body(text):
    document = text.replace('-\n', '').replace('- \n', ' ').replace('\n', ' ').replace('\t', ' ')
    sentences = sent_tokenize(document)
    result =  '<d> <p> ' + ' '.join(['<s> ' + ' '.join(word_tokenize(sentence)).lower() + ' </s>' for sentence in sentences]) + ' </p> </d>'
    return result


#dataset = pytreebank.load_sst()
input_files = ['trees/train.txt','trees/dev.txt','trees/test.txt']
dataset = {key.split('/')[-1].split('.')[0]: pytreebank.import_tree_corpus(key) for key in input_files}
dataset = {key: [(sorted([(len(s),s) for s in t1.to_lines()])[::-1][0][1],t1.label)
 for t1 in dataset[key]] for key in dataset}

dout = []
for key in dataset:
	dout += [{'data':tokenize_body(x[0].lower()),'label':[x[1]],'set':key} for x in dataset[key]]
with open('data.json','w') as fp:
	json.dump(dout, fp,sort_keys=True,indent=4, separators=(',', ': '))
#sentences = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt',sep='\t')
#splits = pd.read_csv('stanfordSentimentTreebank/datasetSplit.txt',sep=',')
#labels = pd.read_csv('stanfordSentimentTreebank/sentiment_labels.txt',sep='|')
#phrases = pd.read_csv('stanfordSentimentTreebank/dictionary.txt',sep='|')
