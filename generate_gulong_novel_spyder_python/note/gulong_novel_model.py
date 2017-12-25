#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:23:48 2017

@author: zhilu
"""
# import module
import pickle
import os
import time
import jieba
import re
import tensorflow as tf
import numpy as np

def load_text(path):
    input_file = os.path.join(path)
    
    with open(input_file, 'r') as f:
        text_data = f.read()
        
    return text_data

def preprocess_save_data(vocab_to_int, int_to_vocab, text):
    
    int_text = [vocab_to_int[word] for word in text]
    
    pickle.dump((int_text, vocab_to_int, int_to_vocab), open('preprocess.p', 'wb'))
    
    
def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))


#create two dict 1. is vocab to integer 2. is integer to vocab
def create_lookup_tables(text):
    #需要的是每个字的idx所以需要set去重
    vocab = set(text)
    
    vocab_to_int = {word : index for index , word in enumerate(vocab)}
    
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab



#40w字
text = load_text('../input/多情剑客无情剑-古龙.txt')

num_words = 200000
text = text[:num_words]

line_train_text = text.split('\n')

print('The total lines in train text is {}'.format(len(line_train_text)))

#delete empty lines

lines_train = [line for line in line_train_text if len(line) > 0]

print('The total lines without blank lines {}'.format(len(lines_train)))

# delete the blank in the begaining or end of the line
#已经用gbk编码里，所以不需要在decode。直接replace就行
lines_train = [line.strip() for line in lines_train]
lines_train = [line.replace('\u3000', u' ') for line in lines_train]
lines_train = [line.replace(' ', u'') for line in lines_train]

raw_text = ''.join(lines_train)
print('before cut there is {0} words'.format(len(raw_text)))

# use jieba cut words
raw_text = jieba.lcut(raw_text)
print('after jieba cut the total words is {}'.format(len(raw_text)))

# 创建两个字典
vocab_to_int, int_to_vocab = create_lookup_tables(raw_text)

# 存储字典和raw_text
preprocess_save_data(vocab_to_int, int_to_vocab, raw_text)
# 重新加载两个词典 以及整篇raw_text对应的int_text
int_text, vocab_to_int, int_to_vocab = load_preprocess()

print('after cut word, the length is {0},a detail example is {1}'.format(len(int_text), int_text[:2]))
print('unique words is {}'.format(len(vocab_to_int)))


#训练循环次数
num_epochs = 300

batch_size = 256

rnn_size = 512

embed_size  = 512

seq_length = 30

#the number of rnn layers
num_layers = 2

learning_rate = 0.001

show_every_n_batches = 20

save_dir = './save'


def get_batches(int_text, batch_size, seq_length):
    
    n_batches = (len(int_text)) // (batch_size * seq_length)
    
    batch_origin =  np.array(int_text[:n_batches * batch_size * seq_length])
    batch_shifted = np.array(int_text[1:n_batches * batch_size * seq_length + 1])
    
    batch_shifted[-1] = batch_origin[0]
    
    #在列方向上切成n_Batches份，由256 × (30*10) 变成 n_batches=10个 256 
    batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, axis=1)
    batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, axis=1)
    
    test = batch_origin.reshape(batch_size, -1)
    print ('origni {}'.format(test.shape))
    batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))
    
    return batches

#得到batched并测试
batched = get_batches(int_text, batch_size, seq_length)
print(batched.shape)
print(batched[0][0].shape)
for batch_idx, (x, y) in enumerate(batched):
    print ('batch_idx: {}, x: {}, y: {}'.format(batch_idx, x.shape, y.shape))
    

#定义inputs
def get_inputs(num_layers, rnn_size):
    
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    init_state = tf.placeholder(tf.float32, [None, num_layers*2*rnn_size], name='init_state')
    
    return inputs, targets, learning_rate, init_state

#定义rnn model
def make_model(rnn_size, num_layers, vocab_size, embed_dim, input_data, ini_state):
    
    #建立基本的rnn net
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=False)
    keep_rate = 0.8
    cell_drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_rate)
    cell = tf.contrib.rnn.MultiRNNCell([cell_drop] * num_layers, state_is_tuple=False)
    
    # 添加embeding层
    embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim]), dtype=tf.float32)
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    #构建完整的rnn net（加入embedding）
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=embed, dtype=tf.float32, initial_state=init_state)
    final_state = tf.identity(final_state, name='final_state')
    
    #构建全连接层
    outputs = tf.reshape(outputs, [-1, rnn_size])
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size]))
    logits = tf.nn.bias_add(tf.matmul(outputs, weights), bias=bias)
    
    return logits, final_state


start = time.time()
train_graph = tf.Graph()
#定义计算图
with train_graph.as_default():
    vocab_size = len(vocab_to_int)
    
    input_text, targets, lr, init_state = get_inputs(num_layers, rnn_size)

    logits, final_state = make_model(rnn_size, num_layers, vocab_size, embed_size, input_text, init_state)

    
    #one hot encoding for target
    labels = tf.one_hot(tf.reshape(targets, [-1]), depth=vocab_size)
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)

    probs = tf.nn.softmax(logits, name='probs')
    
    total_loss = tf.reduce_mean(loss)
    
    train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

#在session中执行计算
with tf.Session(graph=train_graph) as sess:
    
    sess.run(tf.global_variables_initializer())
    
    for epoch_i in range(num_epochs):
        
        state = np.zeros((batch_size, num_layers*2*rnn_size))
        #print('pass run init_state')
        
        for batch_i, (x, y) in enumerate(batched):
            x = np.array(x)
            y = np.array(y)
            feed = {
                input_text : x,
                targets : y,
                lr : learning_rate,
                init_state : state
            }
            train_loss, state, _ = sess.run([total_loss, final_state, train_op], feed)
            
            if(epoch_i * len(batched) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>4}, Batch {:>4}/{}, train_loss {:.3f}'.format(epoch_i, batch_i, len(batched), train_loss))
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model train and saved')
    
    elapsed = (time.time() - start)
    print('used time {}'.format(elapsed))

    
    
def save_params(params):
    pickle.dump(params, open('params.p', 'wb'))
    
def load_params():
    return pickle.load(open('params.p', mode='rb'))

save_params((rnn_size, num_layers, seq_length, save_dir))
