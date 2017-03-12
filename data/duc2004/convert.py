# import struct
import sys
import os
import json
# import sqlite3
# import collections
import re
import random
from nltk.tokenize import sent_tokenize
from glob import glob

random.seed(122956419)

def tokenize_body(text):
  document = text.replace('\n', ' ').replace('\t', ' ')
  sentences = sent_tokenize(document)
  return '<d><p>' + ' '.join(['<s>' + sentence.lower() + '</s>' for sentence in sentences]) + '</p></d>'

docs_folder = "../sources/duc2004/docs"
labels_folder = "../sources/duc2004/eval/models/1/"
output_file = "data.json"
writer = open(output_file, 'w')
writer.write("[\n")

for folder in glob(docs_folder + "/*/"):
  for document_path in glob(folder + "*"):
    _, document_name = os.path.split(document_path)
    # print("folder: " + folder + " file:" + document)
    example = {}
    with open(document_path, 'r') as document_file:
      raw_data = document_file.read()
      text = raw_data.split("<TEXT>")[1].split("</TEXT>")[0].strip()
      example['data'] = tokenize_body(text)
    example['label'] = []
    label_paths = glob(labels_folder + "*" + document_name)
    assert 4 <= len(label_paths) <= 8, "Too many glob hits:  " + str(label_paths)
    for label_path in label_paths:
      with open(label_path, 'r') as label_file:
        example['label'].append(label_file.read().strip())
    example['set'] = random.choices(['train', 'dev', 'test'], weights=[80, 10, 10])[0]
    writer.write(json.dumps(example))
    writer.write(",\n")
writer.write("]")
writer.close()
#
# counter = collections.Counter()
# select = "SELECT title, abstract FROM papers WHERE abstract != 'Abstract Missing';"
# cursor.execute(select)
# results = cursor.fetchall()
# writer = open("data.json", 'w')
# writer.write("[\n")
# for result in results:
#   writer.write("  ") # indent
#   # tokenize etc
#   title = '<d><p><s>' + result[0].lower() + '</s></p></d>'
#   body = result[1].replace('\n', ' ').replace('\t', ' ')
#   sentences = sent_tokenize(body)
#   body = '<d><p>' + ' '.join(['<s>' + sentence.lower() + '</s>' for sentence in sentences]) + '</p></d>'
#   words = " ".join(result).lower().split()
#   counter.update(words)
#   # create and serialize tf_example object
#   example = {}
#   example['data'] = str(body)
#   example['label'] = [str(title)]
#   example['set'] = random.choices(['train', 'dev', 'test'], weights=[80, 10, 10])[0]
#   writer.write(json.dumps(example))
#   writer.write(",\n")
# writer.write("]")
# writer.close()