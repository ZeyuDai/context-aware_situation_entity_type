import sys
import os
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from model import BaseSequenceLabeling, BaseSequenceLabeling_LSTMEncoder, BiLSTMCRF

from sklearn import metrics
import numpy as np
import cPickle
import copy


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

use_cuda = torch.cuda.is_available()
if use_cuda:
    print ("Using GPU!")
else:
    print ("Using CPU!")

def feed_data_cuda(data):
    if use_cuda:
        for X in data:
            for i in range(len(X)):
                X[i] = X[i].cuda()

'''def load_data(para_length_range = [1,10000]):
    print 'Loading Data...'
    with open(os.path.join(os.getcwd(),'data/masc_clause_addposnerembedding_dictformat.pt'),'r') as outfile:
        data = torch.load(outfile)
        outfile.close()

    with open(os.path.join(os.getcwd(),'data/masc_paragraph_dictformat_clause_paralength_list.pkl'),'r') as f:
        masc_paralength_list_dict = cPickle.load(f)
        f.close()

    train_doc_list = []
    test_doc_list = []
    with open('./dataset/MASC_Wikipedia/train_test_split.csv','r') as f:
        for line in f.readlines():
            filename,genre,index,train_test = line.split()
            filename = filename[:-4]
            if train_test == 'train':
                train_doc_list.append(filename)
            elif train_test == 'test':
                test_doc_list.append(filename)

    test_X_eos_list = []
    test_X = []
    test_Y = []
    for filename in test_doc_list:
        doc_x, doc_y = data[filename]
        clause_paralength_list = masc_paralength_list_dict[filename]

        index = -1
        for (sample_X,sample_eos_list,y) in zip(doc_x[0], doc_x[2], doc_y):
            index += 1
            if clause_paralength_list[index] < para_length_range[0] or clause_paralength_list[index] > para_length_range[1]:
                continue

            test_X.append(sample_X)
            test_X_eos_list.append(sample_eos_list)
            test_Y.append(y)

    feed_data_cuda([test_X])
    return test_X,[],test_X_eos_list,test_Y'''

def load_data(para_length_range = [1,10000]):
    print 'Loading Data...'
    with open(os.path.join(os.getcwd(),'data/masc_paragraph_addposnerembedding_dictformat.pt'),'r') as outfile:
        data = torch.load(outfile)
        outfile.close()

    train_doc_list = []
    test_doc_list = []
    with open('./dataset/MASC_Wikipedia/train_test_split.csv','r') as f:
        for line in f.readlines():
            filename,genre,index,train_test = line.split()
            filename = filename[:-4]
            if train_test == 'train':
                train_doc_list.append(filename)
            elif train_test == 'test':
                test_doc_list.append(filename)

    para_length_list = []
    test_X_eos_list = []
    test_X_connective_position_list = []
    test_X = []
    test_Y = []
    for filename in test_doc_list:
        doc_x, doc_y = data[filename]

        for (sample_X,sample_eos_list,sample_connective_position_list,y) in zip(doc_x[0], doc_x[2], doc_x[3], doc_y):
            para_length_list.append(len(sample_eos_list))
            if len(sample_eos_list) < para_length_range[0] or len(sample_eos_list) > para_length_range[1]:
                continue

            test_X.append(sample_X)
            test_X_eos_list.append(sample_eos_list)
            test_X_connective_position_list.append(sample_connective_position_list)
            test_Y.append(y)

    print len(para_length_list)
    print sum(para_length_list) / float(len(para_length_list))
    print 'para length distribution: ' + str(np.unique(para_length_list, return_counts=True))
    feed_data_cuda([test_X])
    return test_X,[],test_X_eos_list,test_X_connective_position_list,test_Y

def process_label(predict_Y,target_Y):
    assert predict_Y.shape == target_Y.shape
    list_1 = []
    list_2 = []

    for i in range(target_Y.shape[0]):
        real_label = target_Y[i,:]
        predict_label = predict_Y[i,:]

        #handle non-label case
        if np.sum(real_label) <= 0:
            continue

        list_1.append(real_label)
        list_2.append(predict_label)

    if len(list_1) > 0:
        real_Y = np.stack(list_1)
        predict_Y = np.stack(list_2)

        return predict_Y,real_Y
    else:
        return None,None

def print_evaluation_result(result):
    predict_Y,target_Y = result[0],result[1]

    print 'Confusion Metric'
    print metrics.confusion_matrix(target_Y,predict_Y)
    print 'Accuracy'
    print metrics.accuracy_score(target_Y, predict_Y)
    print 'Micro Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, average='micro') 
    print 'Macro Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro') 
    print 'Each-class Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, average=None) 

    return metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')[2], result

def evaluate(model,X,Y):
    model.eval()
    X_eos_list = X[1]
    X_connective_position_list = X[2]
    X = X[0]

    predict_Y_list = []
    target_Y_list = []

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        sample_connective_position_list = X_connective_position_list[i]
        target = Y[i]

        #predict = model(sample, sample_eos_list)
        #predict = model(sample, sample_eos_list, sample_connective_position_list)
        predict = model(sample, sample_eos_list, target)

        if use_cuda:
            predict = predict.cpu()

        predict = predict.data.numpy()
        target = target.data.numpy()
        
        predict,target = process_label(predict,target)

        if target is not None:
            predict = np.argmax(predict,axis = 1)
            target = np.argmax(target,axis = 1)

            without_connetive_index = [i for i in range(len(sample_connective_position_list)) if sample_connective_position_list[i]==-1]
            with_connetive_index = [i for i in range(len(sample_connective_position_list)) if sample_connective_position_list[i]!=-1]

            #predict = np.delete(predict, with_connetive_index)
            #target = np.delete(target, with_connetive_index)

            predict_Y_list.append(predict)
            target_Y_list.append(target)

    predict_Y = np.concatenate(predict_Y_list,axis=0)
    target_Y = np.concatenate(target_Y_list,axis=0)
    model.train()
    return print_evaluation_result((predict_Y,target_Y))

def average_result(each_iteration_result_list):
    all_result_list = []

    for each_iteration_result in each_iteration_result_list:
        all_result_list.append(each_iteration_result)

    def average_result_list(result_list):
        predict_Y_list = []
        target_Y_list = []
        for result in result_list:
            predict_Y_list.append(result[0])
            target_Y_list.append(result[1])
        predict_Y = np.concatenate(predict_Y_list,axis=0)
        target_Y = np.concatenate(target_Y_list,axis=0)
        return (predict_Y,target_Y)
    return average_result_list(all_result_list)

batch_size_list = [128]  # fixed 128
hidden_size_list = [300] # fixed 300>200>100>600
dropout_list = [5]  # 3>2>0>5
l2_reg_list = [0.0001]   # fixed 0
nb_epoch_list = [40]
encoder_sentence_embedding_type_list = ['max'] # max > mean > last
sentence_zero_inithidden_list = [False]
crf_decode_method_list = ['viterbi'] #'marginal' < 'viterbi'
loss_function_list = ['likelihood'] # 'likelihood'
optimizer_type_list = ['adam']  # adam > adagrad
num_layers_list = [1] # 1 > 2

parameters_list = []
for num_layers in num_layers_list:
    for sentence_embedding_type in encoder_sentence_embedding_type_list:
        for sentence_zero_inithidden in sentence_zero_inithidden_list:
            for batch_size in batch_size_list:
                for optimizer_type in optimizer_type_list:
                    for hidden_size in hidden_size_list:
                        for nb_epoch in nb_epoch_list:
                            for loss_function in loss_function_list:
                                for crf_decode_method in crf_decode_method_list:
                                    for weight_decay in l2_reg_list:
                                        for dropout in dropout_list:
                                            parameters = {}
                                            parameters['nb_epoch'] = nb_epoch
                                            parameters['sentence_embedding_type'] = sentence_embedding_type
                                            parameters['sentence_zero_inithidden']= sentence_zero_inithidden
                                            parameters['num_layers'] = num_layers
                                            parameters['batch_size'] = batch_size
                                            parameters['hidden_size'] = hidden_size
                                            parameters['crf_decode_method'] = crf_decode_method
                                            parameters['loss_function'] = loss_function
                                            parameters['optimizer_type'] = optimizer_type
                                            parameters['dropout'] = dropout * 0.1
                                            parameters['weight_decay'] = weight_decay
                                            parameters_list.append(parameters)


if __name__ == "__main__":
    test_X,_,test_X_eos_list,test_X_connective_position_list,test_Y = load_data(para_length_range=[1,100000])
    word_embedding_dimension = test_X[0].size(-1)
    number_class = test_Y[0].size(-1)

    parameters = parameters_list[0]
    #stored_model_file = open(os.path.join(os.getcwd(),'./model/masc_clause_addposnerembedding_dictformat_BaseSequenceLabelingLSTMEncoder_eachiteration_adam0.001_l20.0001_senmax_senzeroFalse_layer1_batch128_hidden300_addoutputdropout0.5.pt'),'r')
    #stored_model_file = open(os.path.join(os.getcwd(),'./model/masc_paragraph_addposnerembedding_dictformat_BaseSequenceLabeling_eachiteration_adam0.001_l20.0001_senmax_senzeroFalse_layer1_batch128_hidden300_addoutputdropout0.5.pt'),'r')
    stored_model_file = open(os.path.join(os.getcwd(),'./model/masc_paragraph_addposnerembedding_dictformat_BiLSTMCRF_bestmodel_adam0.001_l20.0001_senmax_senzeroFalse_layer1_batch128_hidden300_addoutputdropout0.5.pt'),'r')
    #stored_model_file = open(os.path.join(os.getcwd(),'./model/masc_paragraph_withoutconnective_addposnerembedding_BaseSequenceLabeling_eachiteration_adam0.001_l20.0001_senmax_senzeroFalse_layer1_batch128_hidden300_addoutputdropout0.5.pt'),'r')


    stored_model_list = torch.load(stored_model_file)
    stored_model_list = [stored_model_list] if type(stored_model_list) != type([]) else stored_model_list
    stored_model_file.close()
    print 'Number of stored model seeds: ' + str(len(stored_model_list))

    overall_best_result = None
    overall_best_model = None
    overall_best_macro = -1
    each_iteration_result_list = []
    each_iteration_macro_Fscore_list = []

    print 'Evaluation: #test_samples= ' + str(len(test_Y))

    for i in range(len(stored_model_list)):
        #model = BaseSequenceLabeling(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                            sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
        
        #model = BaseSequenceLabeling_LSTMEncoder(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                             sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, num_layers = parameters['num_layers'], dropout = parameters['dropout'])

        model = BiLSTMCRF(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                            sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, crf_decode_method = parameters['crf_decode_method'], loss_function = parameters['loss_function'], 
                            num_layers = parameters['num_layers'], dropout = parameters['dropout'])   
        
        if use_cuda:
            model = model.cuda()
        model.load_state_dict(stored_model_list[i])

        print 'Evaluate on all situation entity'
        print '----------------------------------------------------'
        best_macro_Fscore, best_result = evaluate(model,(test_X, test_X_eos_list,test_X_connective_position_list), test_Y)

        each_iteration_result_list.append(best_result)
        each_iteration_macro_Fscore_list.append(best_macro_Fscore)
        if best_macro_Fscore > overall_best_macro:
            overall_best_macro = best_macro_Fscore
            overall_best_result = best_result

    print '--------------------------------------------------------------------------'
    print 'Overall Best Result:'
    print 'Evaluate on all situation entity'
    print '-------------------------------------------------------------------------'
    print_evaluation_result(overall_best_result)

    overall_average_result = average_result(each_iteration_result_list)
    print '-------------------------------------------------------------------------'
    print 'Overall Average Result:'
    print 'Evaluate on all situation entity'
    print '-------------------------------------------------------------------------'
    print_evaluation_result(overall_average_result)
