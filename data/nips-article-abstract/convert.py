# needs python 3.6. sorry

import struct
import sys
import json
import random

import sqlite3
import collections
import re

from nltk.tokenize import sent_tokenize
random.seed(401986418)

def _extract_from_sqlite():
  conn = sqlite3.connect("../sources/nips-papers/database.sqlite")
  cursor = conn.cursor()
  counter = collections.Counter()
  select = "SELECT abstract, paper_text FROM papers WHERE abstract != 'Abstract Missing';"
  cursor.execute(select)
  results = cursor.fetchall()
  dout = []
  for result in results:
    # tokenize etc
    abstract = '<d><p><s>' + result[0].lower() + '</s></p></d>'
    body = result[1].replace('\n', ' ').replace('\t', ' ')
    sentences = sent_tokenize(body)
    body = '<d><p>' + ' '.join(['<s>' + sentence.lower() + '</s>' for sentence in sentences]) + '</p></d>'
    words = " ".join(result).lower().split()
    counter.update(words)
    # create and serialize tf_example object
    example = {}
    example['data'] = str(body)
    example['label'] = [str(abstract)]
    example['set'] = random.choices(['train', 'dev', 'test'], weights=[80, 10, 10])[0]
    dout.append(example)
  with open('data.json','w') as fp:
    json.dump(dout, fp, sort_keys=True, indent=4, separators=(',', ': '))


def main(unused_argv):
  _extract_from_sqlite()


if __name__ == '__main__':
  _extract_from_sqlite()