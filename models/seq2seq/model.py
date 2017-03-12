from load_glove import *
import json
from collections import defaultdict
import numpy as np

GLOVE_LOC = '../../data/glove/glove.6B.50d.txt'
DATASET = '../../data/nips-abstract-title/data.json' #nips-abstract-title #squad
INPUT_MAX = 120
OUTPUT_MAX = 30
MISS_TOKEN = '<UNK>'

words = glove2dict(GLOVE_LOC)
word_counter = defaultdict(int)
VOCAB_MAX = 20000
GLV_DIM = words['the'].shape[0]

def clean(text,clip_n=0):
	res = text.replace('<d>','').replace('<p>','').replace('<s>','').replace('</d>','').replace('</p>','').replace('</s>','')
	
	r2 = []
	for word in res.split():
		if word not in words:
			words[word] = np.array([random.uniform(-0.5, 0.5) for i in range(GLV_DIM)])
	for word in res.split():
		word_counter[word] += 1
	if clip_n > 0:
		return ' '.join(res.split()[:clip_n])
	else:
		return res


with open(DATASET) as fp:
	data = json.load(fp)
	train = [x for x in data if x['set'] == 'train']
	dev = [x for x in data if x['set'] == 'dev']
	test = [x for x in data if x['set'] == 'test']

	train = [(clean(x['data'],INPUT_MAX),clean(x['label'][0],OUTPUT_MAX)) for x in train]
	dev = [(clean(x['data'],INPUT_MAX),clean(x['label'][0],OUTPUT_MAX)) for x in dev]
	test = [(clean(x['data'],INPUT_MAX),clean(x['label'][0],OUTPUT_MAX)) for x in test]

	valid_words = (sorted([(v,k) for k,v in word_counter.iteritems()])[::-1])
	valid_words = [x[1] for x in valid_words[:VOCAB_MAX]]

	#hist([len(x[0].split()) for x in train],100);show()
