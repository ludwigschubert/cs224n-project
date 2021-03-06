
# coding: utf-8

# In[ ]:

from __future__ import print_function
from load_glove import *
import json
from collections import defaultdict
import numpy as np
import tensorflow as tf


# In[ ]:

GLOVE_LOC = '../../data/glove/glove.6B.50d.txt'
DATASET = '../../data/squad/data_s.json' #nips-abstract-title #squad #
INPUT_MAX = 120
OUTPUT_MAX = 25
VOCAB_MAX = 30000

GLV_RANGE = 0.5
LR_DECAY = 100
LR_DECAY_AMOUNT = 0.9
starter_learning_rate = 1e-1
hs = 64
batch_size = 32
PRINT_EVERY = 25
TRAIN_KEEP_PROB = 0.5
TRAIN_EMBEDDING = False


# In[ ]:

words = glove2dict(GLOVE_LOC)
word_counter = defaultdict(int)
GLV_DIM = words['the'].shape[0]

def clean(text,clip_n=0):
    res = text.replace('<d>','').replace('<p>','').replace('<s>','').replace('</d>','').replace('</p>','').replace('</s>','')

    r2 = []
    for word in res.split():
        if word not in words:
            words[word] = np.array([random.uniform(-GLV_RANGE, GLV_RANGE) for i in range(GLV_DIM)])
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

    train = sum([[(clean(x['data'],INPUT_MAX),clean(x['label'][i],OUTPUT_MAX)) for i in xrange(len(x['label']))] for x in train],[])
    dev   = sum([[(clean(x['data'],INPUT_MAX),clean(x['label'][i],OUTPUT_MAX)) for i in xrange(len(x['label']))] for x in dev  ],[])
    test  = sum([[(clean(x['data'],INPUT_MAX),clean(x['label'][i],OUTPUT_MAX)) for i in xrange(len(x['label']))] for x in test ],[])

    valid_words = (sorted([(v,k) for k,v in word_counter.items()])[::-1])
    print(len(valid_words))
    valid_words = [x[1] for x in valid_words[:VOCAB_MAX]] + ['<EOS>','<PAD>','<UNK>','<SOS>']
    unk_idx = valid_words.index('<UNK>')
    vwd = defaultdict(lambda : unk_idx)
    for idx,word in enumerate(valid_words):
        vwd[word] = idx

    initial_matrix = np.array([words[x] for x in valid_words])
    def sent_to_idxs(sentence):
        base =  [vwd[word] for word in sentence.split()]
        sen_len = len(base)
        base =  [vwd['<SOS>']] + base# + [valid_words.index('<EOS>')]
        pad_word = (OUTPUT_MAX-sen_len)
        base = base + pad_word*[vwd['<EOS>']]
        return base,(sen_len,pad_word)
    def sent_to_sum(sentence):
        summed= [words[word] for word in sentence.split()]
        return np.sum(summed,0)/len(sentence)
    train_x = [sent_to_sum(x[0]) for x in train]
    train_y = [sent_to_idxs(x[1])[0] for x in train]
    train_len = [sent_to_idxs(x[1])[1] for x in train]

    dev_x = [sent_to_sum(x[0]) for x in dev]
    dev_y = [sent_to_idxs(x[1])[0] for x in dev]
    dev_len = [sent_to_idxs(x[1])[1] for x in dev]

    test_x = [sent_to_sum(x[0]) for x in test]
    test_y = [sent_to_idxs(x[1])[0] for x in test]
    test_len = [sent_to_idxs(x[1])[1] for x in test]


#train_x[0],train_y[0]


# In[ ]:

tf.reset_default_graph()
global_step = tf.Variable(0, trainable=False)
VOCAB_SIZE = len(valid_words)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,LR_DECAY, LR_DECAY_AMOUNT, staircase=True)

input_placeholder = tf.placeholder(tf.float32,(None,GLV_DIM))
mask_placeholder = tf.placeholder(tf.float32,(None,OUTPUT_MAX))
labels_placeholder = tf.placeholder(tf.int32,(None,OUTPUT_MAX+1))
dropout_rate = tf.placeholder(tf.float32,())

preds = [] # Predicted output at each timestep should go here!

hh0 = tf.get_variable("hh0", shape=[GLV_DIM,hs], initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
hb0 = tf.Variable(tf.constant(0.0, shape=[hs],dtype=tf.float32))
cell = tf.contrib.rnn.GRUCell(hs)

embedding = tf.Variable(initial_matrix,dtype=tf.float32,trainable=TRAIN_EMBEDDING)
looked_up = tf.nn.embedding_lookup(embedding,labels_placeholder)
x = tf.reshape(looked_up,[-1,OUTPUT_MAX+1,GLV_DIM])

U = tf.get_variable("U", shape=(hs,VOCAB_SIZE), initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=(VOCAB_SIZE,), initializer=tf.constant_initializer(0.0))
state = tf.matmul(input_placeholder,hh0) + hb0

outputs, states = tf.nn.dynamic_rnn(cell, x, initial_state=state,dtype=tf.float32)
outputs_batchword = tf.reshape(outputs[:,:OUTPUT_MAX,:],[-1,hs])
out_drop = tf.nn.dropout(outputs_batchword,dropout_rate)
pred_batchword = tf.matmul(out_drop,U) + b2
preds = tf.reshape(pred_batchword,[-1,OUTPUT_MAX,VOCAB_SIZE])

ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds,labels=labels_placeholder[:,1:])
loss = tf.reduce_mean(mask_placeholder * ce)
tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [((tf.clip_by_value(grad, -1., 1.) if grad != None else None), var)  for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs,global_step=global_step)


# In[ ]:

def sample(context_vector):
    sentence = []
    for i in xrange(OUTPUT_MAX):
        x = sent_to_idxs(' '.join(sentence))[0]
        #print(x)
        feed_dict = {
            input_placeholder: context_vector.reshape([1,-1]),
            labels_placeholder: np.array(x).reshape([1,-1]),
            dropout_rate: 1.0
            #mask_placeholder: None
        }
        probs = np.squeeze(preds.eval(feed_dict=feed_dict))
        new_word = valid_words[np.argmax(probs[i,:])]
        if new_word != '<EOS>':
            sentence.append(new_word)
        else:
            break
    return ' '.join(sentence)
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter('./train', sess.graph)
    #saver =  tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    data_size = len(test_x)
    for i in range(data_size*10):
        # cost, label,mask,data_size
        start_idx = (i*batch_size)%data_size
        end_idx = start_idx+batch_size
        #print(train_len[start_idx:end_idx])
        mask = np.array([np.array([1.0]*x[0] + [0.0]*x[1]) for x in train_len[start_idx:end_idx]])
        #print(mask.shape)
        #print(np.array(train_y[start_idx:end_idx]).shape)
        #print(np.array(train_x[start_idx:end_idx]).shape)

        feed_dict = {
            input_placeholder: test_x[start_idx:end_idx],
            labels_placeholder: test_y[start_idx:end_idx],
            mask_placeholder: mask,
            dropout_rate: TRAIN_KEEP_PROB
        }
        _, bl, summary = sess.run([train_step, loss, merged], feed_dict=feed_dict)
        #summary_writer.add_summary(summary, i)
        if i % PRINT_EVERY == 0:
            print(i,bl)
            print('TRAIN_SAMPLE: ',sample(train_x[start_idx]))
            print('TRAIN_LABEL: ',' '.join([x for x in [valid_words[x] for x in train_y[start_idx]] if x not in ['<EOS>','<SOS>']]))
            print()
            index = int(random.random()*10)
            print('DEV_SAMPLE: ',sample(dev_x[index]))
            print('DEV_LABEL: ',' '.join([x for x in [valid_words[x] for x in dev_y[index]] if x not in ['<EOS>','<SOS>']]))
            print('\n')
        #    saver.save(sess, 'model-checkpoint-', global_step=i)
        #    summary_writer.flush()


# In[ ]:




# In[ ]:



