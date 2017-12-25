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
import numpy as np
import tensorflow as tf


def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))

    
def load_params():
    return pickle.load(open('params.p', mode='rb'))

def get_tensors(load_graph):
    inputs = load_graph.get_tensor_by_name('inputs:0')
    
    final_state = load_graph.get_tensor_by_name('init_state:0')
    
    probs = load_graph.get_tensor_by_name('probs:0')
    
    init_state = load_graph.get_tensor_by_name('init_state:0')
    return inputs, final_state, probs, init_state

def predict_word(probabilities, int_to_vocab):
    print('enter predict_word')
    all_words = {idx : prob for idx, prob in enumerate(probabilities)}
    
    sort_all_words = sorted(all_words.items(), key=lambda words:words[1], reverse=True)
    
    Top_five = sort_all_words[:3]
    top_3_prob = []
    top_3_words = []
    for idx, probs in Top_five:
        top_3_prob.append(probs)
        top_3_words.append(int_to_vocab[idx])
        
    print ('the top 3 next words probs is')
    print (top_3_prob)
    print ('the top 3 next words is')
    print (top_3_words)
    
    rand = np.random.randint(0, len(top_3_words))
    
    predict_word = top_3_words[rand]
    print('predict word is {}'.format(predict_word))
    
    return predict_word

_, vocab_to_int, int_to_vocab = load_preprocess()

rnn_size, num_layers, seq_length, load_dir = load_params()


# word length

paragraph_len = 200

first_word = '开始'

loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    #加载保存过的session
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)
    
    #通过名字获取保存的tensor
    input_text, final_state, probs, init_state = get_tensors(loaded_graph)
    
    #准备开始生文本
    gen_sentence = [first_word]
    state = np.zeros((1, num_layers*2*rnn_size))
    
    #开始
    for n in range(paragraph_len):
        print(gen_sentence)
        print('gen length is {}'.format(len(gen_sentence)))
        print('state shape is {}'.format(state.shape))
        temp_input = [[vocab_to_int[word]] for word in gen_sentence[-seq_length:]]
        temp_input = (np.array(temp_input)).reshape(1, -1)
        
        probabilities, state = sess.run([probs, final_state], feed_dict={input_text:temp_input, init_state:state})
        #state = np.reshape(state, (1, -1))
        
        temp_input_length = len(temp_input[0])
        #只需要最后一个字的预测
        print('temp input len is {}'.format(temp_input_length))
        print('probabilities shape {}'.format(probabilities.shape))
        pred_word = predict_word(probabilities[temp_input_length - 1], int_to_vocab)
        print(pred_word)
        gen_sentence.append(pred_word)
        
    novel = ''.join(gen_sentence)
    print(novel)