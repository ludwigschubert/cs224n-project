
# coding: utf-8

# In[ ]:

from load_glove import *
import json
from collections import defaultdict
import numpy as np
import tensorflow as tf


# In[ ]:

GLOVE_LOC = '../../data/glove/glove.6B.50d.txt'
DATASET = '../../data/duc2004/data.json' #nips-abstract-title #squad
INPUT_MAX = 120
OUTPUT_MAX = 30
VOCAB_MAX = 30000

starter_learning_rate = 1e-3
hs = 64


# In[ ]:

words = glove2dict(GLOVE_LOC)
word_counter = defaultdict(int)
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


# In[ ]:

with open(DATASET) as fp:
    data = json.load(fp)
    train = [x for x in data if x['set'] == 'train']
    dev = [x for x in data if x['set'] == 'dev']
    test = [x for x in data if x['set'] == 'test']

    train = [(clean(x['data'],INPUT_MAX),clean(x['label'][0],OUTPUT_MAX)) for x in train]
    dev = [(clean(x['data'],INPUT_MAX),clean(x['label'][0],OUTPUT_MAX)) for x in dev]
    test = [(clean(x['data'],INPUT_MAX),clean(x['label'][0],OUTPUT_MAX)) for x in test]

    valid_words = (sorted([(v,k) for k,v in word_counter.iteritems()])[::-1])
    #print len(valid_words)
    valid_words = [x[1] for x in valid_words[:VOCAB_MAX]] + ['<EOS>','<PAD>','<UNK>','<SOS>']

    initial_matrix = np.array([words[x] for x in valid_words])
    def sent_to_idxs(sentence):
        base = [valid_words.index('<SOS>')] + [valid_words.index(word) for word in sentence.split()]
        base =  base + [valid_words.index('<EOS>')]
        base = base + (OUTPUT_MAX-len(base)+1)*[valid_words.index('<EOS>')]
        return base
    def sent_to_sum(sentence):
        summed= [words[word] for word in sentence.split()]
        return np.sum(summed,0)/len(sentence)
    train_x = [sent_to_sum(x[0]) for x in train]
    train_y = [sent_to_idxs(x[1]) for x in train]

    dev_x = [sent_to_sum(x[0]) for x in dev]
    dev_y = [sent_to_idxs(x[1]) for x in dev]

    test_x = [sent_to_sum(x[0]) for x in test]
    test_y = [sent_to_idxs(x[1]) for x in test]

#train_x[0],train_y[0]


# In[ ]:

tf.reset_default_graph()
global_step = tf.Variable(0, trainable=False)
VOCAB_SIZE = len(valid_words)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,15000, 0.1, staircase=True)

input_placeholder = tf.placeholder(tf.float32,(None,GLV_DIM))
labels_placeholder = tf.placeholder(tf.int32,(None,OUTPUT_MAX+1))

preds = [] # Predicted output at each timestep should go here!

hh0 = tf.get_variable("hh0", shape=[GLV_DIM,hs], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
hb0 = tf.Variable(tf.constant(0.0, shape=[hs],dtype=tf.float32))
cell = tf.contrib.rnn.GRUCell(hs)

embedding = tf.Variable(initial_matrix,dtype=tf.float32)
looked_up = tf.nn.embedding_lookup(embedding,labels_placeholder)
x = tf.reshape(looked_up,[-1,OUTPUT_MAX+1,GLV_DIM])

U = tf.get_variable("U", shape=(hs,VOCAB_SIZE), initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=(VOCAB_SIZE,), initializer=tf.constant_initializer(0.0))
state = tf.matmul(input_placeholder,hh0) + hb0

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
outputs_batchword = tf.reshape(outputs[:,:OUTPUT_MAX,:],[-1,hs])
pred_batchword = tf.matmul(outputs_batchword,U) + b2
preds = tf.reshape(pred_batchword,[-1,OUTPUT_MAX,VOCAB_SIZE])

ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds,labels=labels_placeholder[:,1:])
loss = tf.reduce_mean(ce)


optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [((tf.clip_by_value(grad, -1., 1.) if grad != None else None), var)  for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs,global_step=global_step)


# In[ ]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 32
    data_size = len(train_x)
    for i in range(1000):
        # cost, label,mask,data_size
        start_idx = (i*batch_size)%data_size
        end_idx = start_idx+batch_size

        feed_dict = {
            input_placeholder: train_x[start_idx:end_idx],
            labels_placeholder: train_y[start_idx:end_idx]
        }
        _, bl = sess.run([train_step,loss],feed_dict=feed_dict)
        print i,bl


# In[ ]:



