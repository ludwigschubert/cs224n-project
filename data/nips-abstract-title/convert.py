import struct
import sys
import json

import sqlite3
import collections
import re

from nltk.tokenize import sent_tokenize


def _extract_from_sqlite():
  conn = sqlite3.connect("../nips-papers/database.sqlite")
  cursor = conn.cursor()
  counter = collections.Counter()
  select = "SELECT title, abstract FROM papers WHERE abstract != 'Abstract Missing';"
  cursor.execute(select)
  results = cursor.fetchall()
  writer = open("data.json", 'wb')
  writer.write("[\n")
  for result in results:
    writer.write("  ") # indent
    # tokenize etc
    title = '<d><p><s>' + result[0].lower() + '</s></p></d>'
    body = result[1].decode('utf8').replace('\n', ' ').replace('\t', ' ')
    sentences = sent_tokenize(body)
    body = '<d><p>' + ' '.join(['<s>' + sentence.lower() + '</s>' for sentence in sentences]) + '</p></d>'
    body = body.encode('utf8')
    words = " ".join(result).lower().split()
    counter.update(words)
    # create and serialize tf_example object
    example = {}
    example['data'] = str(body)
    example['label'] = [str(title)]
    writer.write(json.dumps(example))
    writer.write(",\n")
  writer.write("]")
  writer.close()


def main(unused_argv):
  _extract_from_sqlite()


if __name__ == '__main__':
  _extract_from_sqlite()