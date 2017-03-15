import json
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize_body(text):
    document = text.replace('-\n', '').replace('- \n', ' ').replace('\n', ' ').replace('\t', ' ')
    sentences = sent_tokenize(document)
    result =  '<d> <p> ' + ' '.join(['<s> ' + ' '.join(word_tokenize(sentence)).lower() + ' </s>' for sentence in sentences]) + ' </p> </d>'
    return result


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
            sents = context.split('.')
            for sent in sents:
                if len(sent.split()) > 4:
                    dout.append({'data':tokenize_body(sent.lower()),'label':[tokenize_body(sent.lower())],'set':key})

print count_t,count_f
with open('data_s.json','w') as fp:
    json.dump(dout, fp,sort_keys=True,indent=4, separators=(',', ': '))
