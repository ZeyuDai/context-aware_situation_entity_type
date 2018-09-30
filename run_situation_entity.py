import sys
import os
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from model import BaseSequenceLabeling, BaseSequenceLabeling_LSTMEncoder

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

#################################################################

def feed_data_cuda(data):
    if use_cuda:
        for X in data:
            for i in range(len(X)):
                X[i] = X[i].cuda()


'''def load_data():
    print 'Loading Data...'
    outfile = open(os.path.join(os.getcwd(),'data/masc_paragraph_addposnerembedding.pt'),'r')
    data = torch.load(outfile)
    outfile.close()

    train_X,train_Y,test_X,test_Y = data['train_X'] ,data['train_Y'],data['test_X'],data['test_Y']

    train_X_eos_list = train_X[2]
    train_X_label_length_list = train_X[1]
    train_X = train_X[0]

    test_X_eos_list = test_X[2]
    test_X_label_length_list = test_X[1]
    test_X = test_X[0]

    feed_data_cuda([train_X,train_Y,test_X])

    return train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y'''

def load_data():
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

    train_X_eos_list = []
    train_X_label_length_list = []
    train_X = []
    train_Y = []
    for filename in train_doc_list:
        doc_x, doc_y = data[filename]
        train_X += doc_x[0]
        train_X_label_length_list += doc_x[1]
        train_X_eos_list += doc_x[2]
        train_Y += doc_y

    test_X_eos_list = []
    test_X_label_length_list = []
    test_X = []
    test_Y = []
    for filename in test_doc_list:
        doc_x, doc_y = data[filename]
        test_X += doc_x[0]
        test_X_label_length_list += doc_x[1]
        test_X_eos_list += doc_x[2]
        test_Y += doc_y

    feed_data_cuda([train_X,train_Y,test_X])

    return train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y 


#epsilon = 1.0e-6
def calculate_loss(predict,target,criterion):
    #predict = torch.clamp(predict, min=np.log(epsilon), max=np.log(1-epsilon))
    return -torch.dot(predict.view(-1),target.view(-1))

def train(model,X,Y,optimizer,criterion):
    optimizer.zero_grad()
    loss = 0

    X_eos_list = X[1]
    X = X[0]

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        output = model(sample, sample_eos_list)
        loss += calculate_loss(output,target,criterion)

    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    total_norm = nn.utils.clip_grad_norm(model.parameters(), 5.0)

    optimizer.step()
    return loss.data[0]

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
    predict_Y,target_Y,loss = result[0],result[1],result[2]

    print 'Confusion Metric'
    print metrics.confusion_matrix(target_Y,predict_Y)
    print 'Accuracy'
    print metrics.accuracy_score(target_Y, predict_Y)
    print 'loss'
    print loss
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
    X = X[0]

    predict_Y_list = []
    target_Y_list = []
    loss = 0

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        predict = model(sample, sample_eos_list)

        if use_cuda:
            predict = predict.cpu()

        predict = predict.data.numpy()
        target = target.data.numpy()
        
        predict,target = process_label(predict,target)

        if target is not None:
            loss += -np.sum(predict*target)

            predict = np.argmax(predict,axis = 1)
            target = np.argmax(target,axis = 1)

            predict_Y_list.append(predict)
            target_Y_list.append(target)

    predict_Y = np.concatenate(predict_Y_list,axis=0)
    target_Y = np.concatenate(target_Y_list,axis=0)
    model.train()
    return print_evaluation_result((predict_Y,target_Y,loss))


def average_result(each_iteration_result_list):
    all_result_list = []

    for each_iteration_result in each_iteration_result_list:
        all_result_list.append(each_iteration_result)

    def average_result_list(result_list):
        predict_Y_list = []
        target_Y_list = []
        loss = 0
        for result in result_list:
            predict_Y_list.append(result[0])
            target_Y_list.append(result[1])
            loss += result[2]
        predict_Y = np.concatenate(predict_Y_list,axis=0)
        target_Y = np.concatenate(target_Y_list,axis=0)
        return (predict_Y,target_Y,loss)
    
    return average_result_list(all_result_list)

def trainEpochs(model, X, Y, valid_X, valid_Y, batch_size, n_epochs, print_every=1, evaluate_every = 1, optimizer_type = 'adam', weight_decay = 0):
    if optimizer_type == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr = 0.5, weight_decay = weight_decay)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr = 0.01, weight_decay = weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(),lr = 0.001, weight_decay = weight_decay)  #0.001 > 0.0005
    elif optimizer_type == 'adamax':
        optimizer = optim.Adamax(model.parameters(),lr = 0.001, weight_decay = weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = weight_decay)
    else:
        print "optimizer not recommend for the task!"
        sys.exit()

    criterion = nn.NLLLoss()

    X_label_length_list = X[1]
    X_eos_list = X[2]
    X = X[0]
    
    start = time.time()
    random_list = range(len(Y))
    print_loss_total = 0  # Reset every print_every
    best_macro_Fscore = -1
    best_result = None
    best_model = None
    print '----------------------------------------------------'
    print 'Training start: ' + '#training_samples = ' + str(len(Y))
    for epoch in range(1, n_epochs + 1):
        print 'epoch ' + str(epoch) + '/' + str(n_epochs)
        random.shuffle(random_list)

        i = 0
        target_length = 0
        batch = [] 
        while i < len(random_list):
            batch.append(random_list[i])
            target_length += X_label_length_list[random_list[i]]
            i = i + 1

            if target_length >= batch_size or i >= len(random_list):
                batch_X_eos_list = []
                batch_X = []
                batch_Y = []

                for index in batch:
                    batch_X.append(X[index])
                    batch_X_eos_list.append(X_eos_list[index])
                    batch_Y.append(Y[index])

                loss = train(model, (batch_X, batch_X_eos_list), batch_Y, optimizer, criterion)
                print_loss_total += loss

                target_length = 0
                batch = []

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch*1.0/ n_epochs), epoch, epoch*1.0 / n_epochs * 100, print_loss_avg))

        if epoch % evaluate_every == 0:
            print '----------------------------------------------------'
            print 'Step Evaluation: #valid_samples= ' + str(len(valid_Y))
            print 'Evaluate on all situation entity'
            print '----------------------------------------------------'
            tmp_macro_Fscore, tmp_all_result = evaluate(model,valid_X,valid_Y)

            if tmp_macro_Fscore > best_macro_Fscore:
                best_macro_Fscore = tmp_macro_Fscore
                best_result = tmp_all_result
                best_model = copy.deepcopy(model)

    print 'Training completed!'
    return best_macro_Fscore, best_result, best_model



batch_size_list = [128]  # fixed 128
hidden_size_list = [300] # fixed 300>200>100>600
dropout_list = [5]  # 5 > 6 > oher
l2_reg_list = [0.0001]   # fixed 0.0001
nb_epoch_list = [50]
encoder_sentence_embedding_type_list = ['max'] # max > mean > last
sentence_zero_inithidden_list = [False]
optimizer_type_list = ['adam']  # adam > adagrad > other
num_layers_list = [1] # 1 > 2 > (2,1)

parameters_list = []
for num_layers in num_layers_list:
    for sentence_embedding_type in encoder_sentence_embedding_type_list:
        for sentence_zero_inithidden in sentence_zero_inithidden_list:
            for batch_size in batch_size_list:
                for optimizer_type in optimizer_type_list:
                    for hidden_size in hidden_size_list:
                        for nb_epoch in nb_epoch_list:
                            for weight_decay in l2_reg_list:
                                for dropout in dropout_list:
                                    parameters = {}
                                    parameters['nb_epoch'] = nb_epoch
                                    parameters['sentence_embedding_type'] = sentence_embedding_type
                                    parameters['sentence_zero_inithidden']= sentence_zero_inithidden
                                    parameters['num_layers'] = num_layers
                                    parameters['batch_size'] = batch_size
                                    parameters['hidden_size'] = hidden_size
                                    parameters['optimizer_type'] = optimizer_type
                                    parameters['dropout'] = dropout * 0.1
                                    parameters['weight_decay'] = weight_decay
                                    parameters_list.append(parameters)


if __name__ == "__main__":
    train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,_,test_X_eos_list,test_Y = load_data()
    word_embedding_dimension = test_X[0].size(-1)
    number_class = test_Y[0].size(-1)

    macro_F1_list = []
    for parameters in parameters_list:
        overall_best_result = None
        overall_best_model = None
        overall_best_macro = -1
        each_iteration_result_list = []
        each_iteration_best_model_list = []
        each_iteration_macro_Fscore_list = []

        for iteration in range(10):
            #model = BaseSequenceLabeling(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
            #                             sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
            model = BaseSequenceLabeling_LSTMEncoder(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                                         sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
   
            if use_cuda:
                model = model.cuda()

            best_macro_Fscore, best_result, best_model = trainEpochs(model, (train_X,train_X_label_length_list,train_X_eos_list), train_Y, (test_X, test_X_eos_list), test_Y, 
                                                        batch_size = parameters['batch_size'], n_epochs = parameters['nb_epoch'], optimizer_type = parameters['optimizer_type'], weight_decay = parameters['weight_decay'])

            print '----------------------------------------------------'
            print 'Experiment Iteration ' +  str(iteration+1) + ' Evaluation: #test_samples= ' + str(len(test_Y))
            print 'Evaluate on all situation entity'
            print '----------------------------------------------------'
            print_evaluation_result(best_result)


            each_iteration_result_list.append(best_result)
            each_iteration_best_model_list.append(best_model.state_dict())
            each_iteration_macro_Fscore_list.append(best_macro_Fscore)
            if best_macro_Fscore > overall_best_macro:
                overall_best_macro = best_macro_Fscore
                overall_best_result = best_result
                overall_best_model = best_model

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
        macro_F1_list.append(print_evaluation_result(overall_average_result)[0])

        torch.save(each_iteration_best_model_list, './model/masc_paragraph_addposnerembedding_dictformat_BaseSequenceLabelingLSTMEncoder_eachiteration_adam0.001_'+ 'l2'+ str(parameters['weight_decay']) + '_sen' + parameters['sentence_embedding_type']+ '_senzero'+ str(parameters['sentence_zero_inithidden'])+ '_layer' +  str(parameters['num_layers']) + '_batch' + str(parameters['batch_size']) +  '_hidden'+ str(parameters['hidden_size'])  + '_addoutputdropout' + str(parameters['dropout']) + '.pt')
        #torch.save(overall_best_model.state_dict(), './model/masc_paragraph_addposnerembedding_dictformat_BaseSequenceLabeling_bestmodel_adam0.001_'+ 'l2'+ str(parameters['weight_decay']) + '_sen' + parameters['sentence_embedding_type']+ '_senzero'+ str(parameters['sentence_zero_inithidden'])+ '_layer' +  str(parameters['num_layers']) + '_batch' + str(parameters['batch_size']) +  '_hidden'+ str(parameters['hidden_size'])  + '_addoutputdropout' + str(parameters['dropout']) + '.pt')

        print 'Situation entity classification Macro F_score std:' + str(np.std(each_iteration_macro_Fscore_list))

        print str(parameters)
        sys.stdout.flush()

    print '-------------------------------------------------------------------------'
    print '-------------------------------------------------------------------------'
    print '-------------------------------------------------------------------------'
    for i in range(len(parameters_list)):
        print str(parameters_list[i]) + '       macro F-1:  ' + str(macro_F1_list[i]) + '\n'