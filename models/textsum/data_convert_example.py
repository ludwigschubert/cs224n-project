"""Example of Converting TextSum model data.
Usage:
python data_convert_example.py --command binary_to_text --in_file data/data --out_file data/text_data
python data_convert_example.py --command text_to_binary --in_file data/text_data --out_file data/binary_data
python data_convert_example.py --command binary_to_text --in_file data/binary_data --out_file data/text_data2
diff data/text_data2 data/text_data
"""

import struct
import sys

import tensorflow as tf
from tensorflow.core.example import example_pb2

import sqlite3
import collections
import re

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('db', '', 'path to db file')
tf.app.flags.DEFINE_string('out_file', '', 'path to output file')
tf.app.flags.DEFINE_string('vocab_file', '', 'path to vocab output file')


def _extract_from_sqlite():
  conn = sqlite3.connect(FLAGS.db)
  cursor = conn.cursor()
  counter = collections.Counter()
  features = ('title', 'abstract')
  select = "SELECT title, abstract FROM papers WHERE abstract != 'Abstract Missing';"
  cursor.execute(select)
  results = cursor.fetchall()
  writer = open(FLAGS.out_file, 'wb')
  for result in results:
    tf_example = example_pb2.Example()
    for index, feature in enumerate(features):
      result = map(lambda x: x.lower(), result)
      tf_example.features.feature[feature].bytes_list.value.extend([str(result[index])])
      words = re.split('[^a-zA-Z]', " ".join(result))
      counter.update(words)
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))
  writer.close()
  # write vocab file
  with open(FLAGS.vocab_file, 'w') as writer:
    for word, count in counter.most_common(100000):
      writer.write(word + ' ' + str(count) + '\n')
    writer.write('<s> 0\n')
    writer.write('</s> 0\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')


def main(unused_argv):
  assert FLAGS.db and FLAGS.out_file and FLAGS.vocab_file
  _extract_from_sqlite()


if __name__ == '__main__':
  tf.app.run()
