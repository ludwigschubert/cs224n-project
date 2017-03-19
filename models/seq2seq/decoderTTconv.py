
# coding: utf-8

from __future__ import print_function
from load_glove import *
import json
from collections import defaultdict
import numpy as np
import tensorflow as tf
from sys import argv
import os
import argparse
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('--runmode', dest='runmode', choices=["train", "test"], default="train")
parser.add_argument('--dataset_name', dest='dataset_name', type=str, default="duc2004")
parser.add_argument('--trained_on', dest='trained_on', type=str, default="duc2004")
parser.add_argument('--cnn', dest='cnn', type=int, default=0)
parser.add_argument('--train_embedding', dest='train_embedding', type=int, default=0)
parser.add_argument('--output_root', dest='output_root', type=str, default="")
parser.add_argument('--evaluation_root', dest='evaluation_root', type=str, default="../../evaluation")
args = parser.parse_args()

log_components = ["train"]
if args.runmode == "train":
    log_components += [args.dataset_name]
elif args.runmode == "test":
    log_components += [args.trained_on]
if args.cnn:
    log_components += ["cnn"]
if args.train_embedding:
    log_components += ["train_embedding"]
LOGDIR = "-".join(log_components)
if args.output_root != "":
    LOGDIR = os.path.join(args.output_root, LOGDIR)
dataset_file = os.path.join("../../data", args.dataset_name, "data.json")
print("Runmode %s on dataset %s" % (args.runmode, args.dataset_name))

if args.runmode == "test":
    predictions_dir_name = "_".join(["seq2seq", LOGDIR, args.trained_on, args.dataset_name])
    predictions_dir_path = os.path.join(args.evaluation_root, predictions_dir_name)
    predictions_file_path = os.path.join(predictions_dir_path, "prediction.json.gz")

GLOVE_LOC = '../../data/glove/glove.6B.50d.txt'

INPUT_MAX = 150
OUTPUT_MAX = 20
VOCAB_MAX = 2000

GLV_RANGE = 0.5
LR_DECAY_AMOUNT = 0.8
starter_learning_rate = 1e-3
hs = 8

batch_size = 32
PRINT_EVERY = 10
CHECKPOINT_EVERY = 5000
TRAIN_KEEP_PROB = 0.5
TRAIN_EMBEDDING = args.train_embedding
KERNEL_SIZE = 7
if args.runmode == "train":
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
if args.runmode == "test":
    if not os.path.exists(predictions_dir_path):
        os.makedirs(predictions_dir_path)

words = glove2dict(GLOVE_LOC)
word_counter = defaultdict(int)
GLV_DIM = words['the'].shape[0]
not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
translate_table = dict((ord(char), None) for char in not_letters_or_digits)
def clean(text,clip_n=0):
    res = text.replace('<d>','').replace('<p>','').replace('<s>','').replace('</d>','').replace('</p>','').replace('</s>','').translate(translate_table)
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

from collections import defaultdict
with open(dataset_file) as fp:
    data = json.load(fp)
    train_o = [x for x in data if x['set'] == 'train']
    dev_o = [x for x in data if x['set'] == 'dev']
    test_o = [x for x in data if x['set'] == 'test']

    train = sum([[(clean(x['data'],INPUT_MAX), clean(x['label'][i],OUTPUT_MAX),idx) for i in xrange(len(x['label']))] for idx,x in enumerate(train_o)],[])
    dev   = sum([[(clean(x['data'],INPUT_MAX), clean(x['label'][i],OUTPUT_MAX),idx) for i in xrange(len(x['label']))] for idx,x in enumerate(dev_o)  ],[])
    test  = sum([[(clean(x['data'],INPUT_MAX), clean(x['label'][i],OUTPUT_MAX),idx) for i in xrange(len(x['label']))] for idx,x in enumerate(test_o) ],[])

    valid_words = (sorted([(v,k) for k,v in word_counter.items()])[::-1])
    print(len(valid_words))
    valid_words = ['<PAD>'] + [x[1] for x in valid_words[:VOCAB_MAX]] + ['<EOS>','<UNK>','<SOS>']
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
        if pad_word == 0:
            return base,(sen_len,pad_word) 
        else:
            return base,(sen_len+1,pad_word-1)
    def sent_to_idxs_nopad(sentence):
        base =  [vwd[word] for word in sentence.split()]
        return base
    random.shuffle(train)
    train_x = [sent_to_idxs_nopad(x[0]) for x in train]
    train_y = [sent_to_idxs(x[1])[0] for x in train]
    train_len = [sent_to_idxs(x[1])[1] for x in train]

    dev_x = [sent_to_idxs_nopad(x[0]) for x in dev]
    dev_y = [sent_to_idxs(x[1])[0] for x in dev]
    dev_len = [sent_to_idxs(x[1])[1] for x in dev]

    test_x = [sent_to_idxs_nopad(x[0]) for x in test]
    test_y = [sent_to_idxs(x[1])[0] for x in test]
    test_len = [sent_to_idxs(x[1])[1] for x in test]


def try_restoring_checkpoint(session, saver):
    print('trying to restore checkpoints...')
    try:
      ckpt_state = tf.train.get_checkpoint_state(LOGDIR)
    except tf.errors.OutOfRangeError as e:
      print('Cannot restore checkpoint: ', e)
      exit(1)

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      print('No model to eval yet at ', LOGDIR)
      return

    print('Loading checkpoint ', ckpt_state.model_checkpoint_path)
    saver.restore(session, ckpt_state.model_checkpoint_path)
    print('...loaded.')

tf.reset_default_graph()
global_step = tf.Variable(0, trainable=False)
VOCAB_SIZE = len(valid_words)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, len(train_x)/batch_size, LR_DECAY_AMOUNT, staircase=True)

input_placeholder = tf.placeholder(tf.int32)
mask_placeholder = tf.placeholder(tf.bool,(None,OUTPUT_MAX))
labels_placeholder = tf.placeholder(tf.int32,(None,OUTPUT_MAX+1))
dropout_rate = tf.placeholder(tf.float32,())

embedding = tf.Variable(initial_matrix,dtype=tf.float32,trainable=TRAIN_EMBEDDING)
input_embed = tf.nn.embedding_lookup(embedding,input_placeholder)
input_summed= tf.reduce_mean(input_embed,1)

#out_range = tf.range(0,OUTPUT_MAX,dtype=tf.float32)
out_range = tf.linspace(0.0,1.0,OUTPUT_MAX)
out_range = tf.reshape(tf.tile(out_range,[tf.shape(input_placeholder)[0]]),[-1,OUTPUT_MAX,1])
dupe_state = tf.reshape(tf.tile(input_summed,[1,OUTPUT_MAX]),[-1,OUTPUT_MAX,GLV_DIM])
input_data  = tf.concat([out_range,dupe_state],2)

W1 = tf.get_variable("W1", shape=[KERNEL_SIZE,GLV_DIM+1,hs], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.constant(0.0, shape=[hs]))
W2 = tf.get_variable("W2", shape=[KERNEL_SIZE,GLV_DIM+1+hs,hs*2], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.constant(0.0, shape=[hs*2]))
W3 = tf.get_variable("W3", shape=[KERNEL_SIZE,GLV_DIM+1+hs*3,3*hs], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.constant(0.0, shape=[3*hs]))

W4 = tf.get_variable("W4", shape=[KERNEL_SIZE,hs*3,3*hs], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.constant(0.0, shape=[3*hs]))
W5 = tf.get_variable("W5", shape=[KERNEL_SIZE,hs*3,3*hs], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.constant(0.0, shape=[3*hs]))
W6 = tf.get_variable("W6", shape=[KERNEL_SIZE,6*hs,VOCAB_SIZE], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.constant(0.0, shape=[VOCAB_SIZE]))

h_conv1 = tf.nn.elu(tf.nn.conv1d(input_data, W1, stride=1, padding='SAME') + b1)
h_conv2 = tf.nn.elu(tf.nn.conv1d(tf.concat([h_conv1,input_data],2)           , W2, stride=1, padding='SAME') + b2)
h_conv3 = tf.nn.elu(tf.nn.conv1d(tf.concat([h_conv1,h_conv2,input_data],2)   , W3, stride=1, padding='SAME') + b3)
h_conv4 = tf.nn.elu(tf.nn.conv1d(h_conv3                                     , W4, stride=1, padding='SAME') + b4)
h_conv5 = tf.nn.elu(tf.nn.conv1d(h_conv4                                     , W5, stride=1, padding='SAME') + b5)
h_conv6 = tf.nn.elu(tf.nn.conv1d(tf.concat([h_conv3,h_conv4],2)              , W6, stride=1, padding='SAME') + b6)

preds = h_conv6

ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds,labels=labels_placeholder[:,1:])
loss = tf.reduce_mean(tf.boolean_mask(ce,mask_placeholder))
tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(loss)
#capped_gvs = [((tf.clip_by_value(grad, -1., 1.) if grad != None else None), var)  for grad, var in gvs]
train_step = optimizer.apply_gradients(gvs,global_step=global_step)


def sample(context_vector):
    sentence = []

    feed_dict = {
        input_placeholder: np.array(context_vector).reshape([1,-1]),
        #labels_placeholder: np.array(x).reshape([1,-1]),
        dropout_rate: 1.0
    }
    probs = np.squeeze(preds.eval(feed_dict=feed_dict))
    for idx in np.argmax(probs,axis=1):
        #print(probs.shape)
        new_word = valid_words[idx]
        if new_word != '<EOS>':
            sentence.append(new_word)
        else:
            break
    return ' '.join(sentence)
def sample_batch(context_vectors):
    sizes = [len(x) for x in context_vectors]
    mat = np.zeros(shape=(len(context_vectors),INPUT_MAX))
    for idx,row in enumerate(context_vectors):
        mat[idx,:sizes[idx]] = np.array(row)
    num_sent = np.array(context_vectors).shape[0]
    sentences = [[] for _ in xrange(num_sent)]

    for i in xrange(OUTPUT_MAX):
        x = [sent_to_idxs(' '.join(s))[0] for s in sentences]
        feed_dict = {
            input_placeholder: mat.reshape([num_sent,-1]),
            labels_placeholder: np.array(x).reshape([num_sent,-1]),
            dropout_rate: 1.0
        }
        probs = preds.eval(feed_dict=feed_dict) # batch,word,vocab
        for batch_i,batch_prob in enumerate(probs):
            new_word = valid_words[np.argmax(batch_prob[i,:])]
            sentences[batch_i].append(new_word)
    stops = [sentence.index('<EOS>') if '<EOS>' in sentence else OUTPUT_MAX for sentence in sentences]
    return [' '.join(sentence[:maxe]) for maxe,sentence in zip(stops,sentences)]

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    saver =  tf.train.Saver()
    try_restoring_checkpoint(sess, saver)
    data_size = len(train_x)
    if args.runmode == "train":

        for i in range(data_size*10):
            start_idx = (i*batch_size)%data_size
            end_idx = start_idx+batch_size
            mask = np.array([np.array([True]*x[0] + [False]*x[1]) for x in train_len[start_idx:end_idx]])
            train_sizes = [len(x) for x in train_x[start_idx:end_idx]]
            train_mat = np.zeros(shape=(len(train_sizes),max(train_sizes)))
            for idx,row in enumerate(train_x[start_idx:end_idx]):
                train_mat[idx,:train_sizes[idx]] = np.array(row)
            feed_dict = {
                input_placeholder: train_mat,
                labels_placeholder: train_y[start_idx:end_idx],
                mask_placeholder: mask,
                dropout_rate: TRAIN_KEEP_PROB
            }
            _, bl, summary = sess.run([train_step, loss, merged], feed_dict=feed_dict)
            if args.runmode == "train":
                summary_writer.add_summary(summary, i)
            if i % PRINT_EVERY == 0:
                print(i,bl)
                print('TRAIN_SAMPLE: ',sample(train_x[start_idx]))
                print('TRAIN_LABEL1: ',' '.join([x for x in [valid_words[x] for x in train_y[start_idx]] if x not in ['<EOS>','<SOS>']]))
                print('TRAIN_LABEL2: ',' '.join([x for x,m in zip([valid_words[x] for x in train_y[start_idx][1:]],mask[0]) if m]))
                print('TRAIN_LABEL2: ',' '.join([x for x,m in zip([valid_words[x] for x in train_y[start_idx][1:]],mask[0]) if True]))

                index = int(random.random()*(len(dev_y)-1))
                print('DEV_SAMPLE: ',sample(dev_x[index]))
                print('DEV_LABEL: ',' '.join([x for x in [valid_words[x] for x in dev_y[index]] if x not in ['<EOS>','<SOS>']]))
                print('\n')
            if i != 0 and i %2000*len(train_x)/batch_size == 0:
                print("Saving checkpoint...")
                saver.save(sess, os.path.join(LOGDIR, 'model-checkpoint-'), global_step=i)
    if args.runmode == "test":
        batch_size = 2048
        orig_file = train_o
        x_file = train_x
        src_file = train

        print("Running predictions for %d data points..." % len(x_file))
        predictions = []
        seen_data = {}
        for batch_i in xrange(0,len(x_file),batch_size):
            evaluation_data = x_file[batch_i:batch_i+batch_size]
            prediction_results = sample_batch(evaluation_data)
            for i, prediction in enumerate(prediction_results):
                if src_file[i][2] not in seen_data:
                    orig_data = orig_file[src_file[i][2]]
                    orig_data['prediction'] = "<d> <p> <s> " + prediction + " </s> </p> </d>"
                    predictions.append(orig_data)
                    seen_data[src_file[i][2]] = 1
        print("Done, writing json.gz file...")
        with gzip.open(predictions_file_path, 'w') as predictions_file:
            json.dump(predictions, predictions_file, sort_keys=True, indent=4, separators=(',', ': '))
        print("All done.")
