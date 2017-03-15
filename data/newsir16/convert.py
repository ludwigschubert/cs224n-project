import json
import gzip
import random
from nltk.tokenize import sent_tokenize, word_tokenize

def tokenize_body(text):
        document = text.replace('-\n', '').replace('- \n', ' ').replace('\n', ' ').replace('\t', ' ')
        sentences = sent_tokenize(document)
        result =  '<d> <p> ' + ' '.join(['<s> ' + ' '.join(word_tokenize(sentence)).lower() + ' </s>' for sentence in sentences]) + ' </p> </d>'
        return result


with gzip.open('signalmedia-1m.jsonl.gz') as fp:
    dout = []
    for line in fp:
	news_article = json.loads(line)
        title = news_article['title'].lower()
        if news_article['media-type'] != 'News' or '?' in title or '(' in title or '"' in title or 'tips' in title or "..." in title or 'hacks' in title or '1' in title:
            continue
	#print news_article['title'], '\n',len(news_article['content']),news_article['source'],news_article['media-type']
	label = 'train'
	if random.random() > 0.8:
	    if random.random() > 0.5:
		label = 'dev'
	    else:
		label = 'test'
                print '\r',len(dout),
                if len(dout) > 10000:
                    break
	dout.append({'data':tokenize_body(news_article['content'].lower()),'label':[tokenize_body(news_article['title'])],'set':label})

with open('data.json','w') as fp:
        json.dump(dout, fp,sort_keys=True,indent=4, separators=(',', ': '))

