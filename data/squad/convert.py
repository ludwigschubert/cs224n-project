import json

files = ['train-v1.1.json','dev-v1.1.json']
dataset = {}
for file in files:
	with open(file,'rb') as fp:
		dataset[file.split('-')[0]] = json.load(fp)['data']

#dataset['dev'][0]
#['paragraphs'][0] #['context'] | (['qas'] [0] (['question'] | answers[0] ['text'] 
count_t = 0
count_f = 0
dout = []
for key in dataset:
	for story in dataset[key]:
		for para in story['paragraphs']:
			context = para['context']
			labels = []
			for qas in para['qas']:
				q = qas['question'].lower().replace('?','.')
				sa = sorted([(len(a['text']),a['text'].lower()) for a in qas['answers']])[0][1]
				if 'who' in q :
					count_t +=1
					labels.append(q.replace('who',sa))
				#elif 'what was' in q:
				#	print sa, '|', q.replace('what',sa)
				#	count_t +=1
				else:
					count_f +=1
			if len(labels) >0:
				dout.append({'data':context.lower(),'label':labels,'set':key})

print count_t,count_f 
with open('data.json','w') as fp:
	json.dump(dout, fp,sort_keys=True,indent=4, separators=(',', ': '))