#!/usr/local/bin/python

"""
SQLite analysis template
2017-03-05 Ludwig Schubert
"""

"""
Uncomment to import data tools
"""
import sqlite3
import collections
import pickle

import numpy as np
import matplotlib.pyplot as plt

"""
Uncomment to import text tools
"""
# import re
from nltk.tokenize import sent_tokenize, word_tokenize

db = "/Users/ludwig/Code/cs224n-project/data/nips-papers/database.sqlite"
ref_file = "/Users/ludwig/Documents/2017-03-05-textsum/ref1488756998"
out_file = "/Users/ludwig/Documents/2017-03-05-textsum/decode1488756998"

def analyse():
  connection = sqlite3.connect(db)
  cursor = connection.cursor()
  counter = collections.Counter()
  out_counter = collections.Counter()
  with open(ref_file, 'r') as refs:
    with open(out_file, 'r') as outs:
      for ref in refs:
        _, title     =             ref.rstrip('\n').split('=')
        _, out_title = outs.readline().rstrip('\n').split('=')
        query = "SELECT DISTINCT abstract FROM papers WHERE title LIKE ? LIMIT 1"
        cursor.execute(query, (title,))
        abstract = cursor.fetchone()
        if abstract:
          abstract = abstract[0].replace('\n', ' ').replace('\t', ' ')
          abstract_words = [word.lower().encode('utf8') for sentence in sent_tokenize(abstract) for word in word_tokenize(sentence)]
          title_words = word_tokenize(title)
          out_title_words = word_tokenize(out_title)
          title_indexes     = [int(i * (100.0/len(abstract_words))) for i, word in enumerate(abstract_words) if word in title_words]
          out_title_indexes = [int(i * (100.0/len(abstract_words))) for i, word in enumerate(abstract_words) if word in out_title_words]
          counter.update(title_indexes)
          out_counter.update(out_title_indexes)
        else:
          print("Could not find: ", title)
  print("Counter")
  counter_diagram(counter, "counter")
  print("Out Counter")
  counter_diagram(out_counter, "out_counter")

def counter_diagram(a_counter, name):
  labels, values = zip(*a_counter.items())
  indexes = np.arange(len(labels))
  width = 1
  plt.xlabel('Position of title word in abstract in percent of abstract length')
  plt.ylabel('Count')
  plt.title('Histogram of positions of words in abstracts which appearing in title')
  plt.bar(indexes, values, width, edgecolor = "none")
  plt.axis([0, 100, 0, 1750])
  # plt.xticks(indexes + width * 0.5, labels)
  fig = plt.gcf()
  fig.savefig(name + '.pdf')
  # clear plot for next draw
  plt.clf()
  plt.cla()

def main(argv):
  analyse()

if __name__ == '__main__':
  main([])
