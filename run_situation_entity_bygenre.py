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

######################################################################

def feed_data_cuda(data):
    if use_cuda:
        for X in data:
            for i in range(len(X)):
                X[i] = X[i].cuda()

def feed_data_cpu(data):
    if use_cuda:
        for X in data:
            for i in range(len(X)):
                X[i] = X[i].cpu()

genre_list = ['blog','email','essays','ficlets','fiction','govt-docs','jokes','journal','letters','news','technical','travel','wiki']
def load_data(given_fold_list = None, learning_curve = False):
    print 'Loading Data...'
    outfile = open(os.path.join(os.getcwd(),'data/masc_paragraph_addposnerembedding_dictformat.pt'),'r')
    data = torch.load(outfile)
    outfile.close()

    if given_fold_list is None:
        fold_doc_list = [[] for i in range(len(genre_list))]
        with open('./dataset/MASC_Wikipedia/train_test_split.csv','r') as f:
            for line in f.readlines():
                filename,genre,index,train_test = line.split()
                filename = filename[:-4]

                if train_test != 'train':
                    continue

                if genre in genre_list:
                    fold_doc_list[genre_list.index(genre)].append(filename)
            f.close()
    else:
        fold_doc_list = given_fold_list

    fold_list = []
    if not learning_curve:
        for i in range(len(fold_doc_list)):
            train_X_eos_list = []
            train_X_label_length_list = []
            train_X = []
            train_Y = []
            test_X_eos_list = []
            test_X_label_length_list = []
            test_X = []
            test_Y = []

            for j in range(len(fold_doc_list)):
                if j != i:
                    for filename in fold_doc_list[j]:
                        doc_x, doc_y = data[filename]
                        train_X += doc_x[0]
                        train_X_label_length_list += doc_x[1]
                        train_X_eos_list += doc_x[2]
                        train_Y += doc_y
                else:
                    for filename in fold_doc_list[j]:
                        doc_x, doc_y = data[filename]
                        test_X += doc_x[0]
                        test_X_label_length_list += doc_x[1]
                        test_X_eos_list += doc_x[2]
                        test_Y += doc_y
            fold_list.append([train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y])
    else:
        test_doc_list = []
        with open('./dataset/MASC_Wikipedia/train_test_split.csv','r') as f:
            for line in f.readlines():
                filename,genre,index,train_test = line.split()
                filename = filename[:-4]
                if train_test == 'test':
                    test_doc_list.append(filename)
            f.close()

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

        for i in range(len(fold_doc_list)):
            train_X_eos_list = []
            train_X_label_length_list = []
            train_X = []
            train_Y = []

            for filename in fold_doc_list[i]:
                doc_x, doc_y = data[filename]
                train_X += doc_x[0]
                train_X_label_length_list += doc_x[1]
                train_X_eos_list += doc_x[2]
                train_Y += doc_y              
            fold_list.append([train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y])
    return fold_list


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
nb_epoch_list = [40]
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
    fold_list = load_data()

    # Cross-validation on training set:
    #fold_list = load_data( given_fold_list = [['letters_audubon2', 'email_175814', 'news_wsj_0106', 'ficlets_1400', 'wiki_choir', 'wiki_ayoreo_people', 'email_174124', 'letters_113CWL017', 'wiki_color_blindness', 'jokes_jokes5', 'email_lists-003-2152883', 'news_20000419_apw_eng-NEW', 'wiki_hells_angels', 'blog_blog-varsity-athletics', 'email_lists-034-10077455', 'travel_HistoryLasVegas', 'wiki_single-party_state', 'letters_116CUL034', 'email_ENRON-pearson-email-25jul02', 'wiki_mantis_shrimp', 'email_lists-003-2171003', 'wiki_left-wing_politics', 'blog_How_soon-Fans', 'fiction_lw1', 'email_173906', 'letters_112C-L013', 'wiki_human_rights', 'news_A1.E2-NEW', 'news_NYTnewswire2'], ['letters_audubon1', 'news_wsj_0161', 'blog_Italy', 'email_53536', 'wiki_bandidos', 'email_219257', 'wiki_bari_people', 'technical_1471-213X-1-1', 'email_49059', 'wiki_Archimedes_screw', 'letters_116CUL032', 'email_53555', 'wiki_herbalism', 'letters_110CYL072', 'jokes_jokes4', 'wiki_stratocracy', 'news_wsj_0073', 'email_54261', 'wiki_despotism', 'email_lists-046-11485666', 'letters_117CWL008', 'journal_VOL15_3', 'wiki_leafGen', 'news_NYTnewswire7', 'jokes_jokes3', 'email_52201', 'news_wsj_0136', 'wiki_chemistry', 'essays_anth_essay_4'], ['news_20000424_nyt-NEW', 'letters_118CWL049', 'wiki_american_football', 'email_lists-003-2205935', 'govt-docs_fcic_final_report_conclusions', 'travel_HistoryGreek', 'wiki_yuqui_people', 'email_lists-003-2121270', 'letters_110CYL067', 'wiki_pintupi', 'blog_Fermentation_HR5034', 'email_54536', 'essays_Black_and_white', 'wiki_democracyGen', 'email_175448', 'jokes_jokes8', 'letters_110CYL071', 'email_8885', 'wiki_bagpipes', 'news_wsj_0159', 'wiki_okapi', 'letters_115CVL037', 'email_211402', 'news_wsj_0124', 'wiki_shoebill', 'letters_115CVL036', 'email_lists-003-2144868', 'news_20000410_nyt-NEW', 'news_NYTnewswire6'], ['email_176581', 'letters_AMC2', 'news_20020731-nyt', 'ficlets_1401', 'wiki_bloods', 'email_221197', 'letters_112C-L016', 'wiki_veterinary_medicine', 'wiki_papyrus', 'email_52713', 'letters_alumnifund1', 'email_21257', 'wiki_neurology', 'blog_How_soon-Lebron-James', 'govt-docs_chapter-10', 'email_9085', 'wiki_zebras', 'wiki_parapsychology', 'technical_1468-6708-3-1', 'news_wsj_0150', 'jokes_jokes11', 'letters_115CVL035', 'email_211401', 'news_wsj_0127', 'wiki_Alexander_Stepanovich_Popov', 'wiki_Harry_Coover', 'email_enron-thread-159550', 'news_NYTnewswire8', 'fiction_Nathans_Bylichka'], ['wiki_venusflytrapGen', 'email_230685', 'blog_Anti-Terrorist', 'ficlets_1402', 'letters_118CWL050', 'news_wsj_0027', 'wiki_capitalism', 'email_219122', 'letters_117CWL009', 'wiki_basketball', 'email_lists-034-10062451', 'wiki_go_fish', 'email_lists-034-10066763', 'letters_114CUL057', 'jokes_jokes2', 'wiki_yodeling', 'email_lists-003-2114716', 'wiki_occultism', 'journal_Article247_327', 'news_wsj_0151', 'travel_IntroDublin', 'letters_defenders5', 'wiki_clavichord', 'news_wsj_0120', 'email_lists-003-2173878', 'wiki_gusli', 'email_218920', 'news_NYTnewswire9'], ['email_234267', 'ficlets_1399', 'wiki_consumerlawsGen', 'letters_guidedogs1', 'email_12174', 'letters_602CZL285', 'news_20000415_apw_eng-NEW', 'wiki_dictatorship', 'email_173252', 'blog_blog-new-years-resolutions', 'wiki_philosophy_of_mind', 'email_lists-003-2125109', 'letters_114CUL058', 'wiki_buddhismGen', 'jokes_jokes1', 'wiki_volleyball', 'email_54263', 'journal_Article247_328', 'wiki_baseball', 'news_wsj_0152', 'news_wsj_0160', 'wiki_sloth', 'email_12030', 'blog_Fermentation_Eminent-Domain', 'letters_112C-L015', 'travel_HistoryJerusalem', 'news_wsj_0132', 'email_lists-003-2129640'], ['email_234783', 'letters_att2', 'technical_1471-2091-2-9', 'wiki_Louis_Braille', 'email_12176', 'wiki_patolli', 'govt-docs_Env_Prot_Agency-nov1', 'essays_Homosexuality', 'news_wsj_0165', 'letters_110CYL068', 'wiki_Robert_Koch', 'email_9191', 'letters_114CUL059', 'wiki_free_will', 'email_9066', 'blog_Acephalous-Internet', 'email_54262', 'wiki_Joseph_Swan', 'wiki_hinduism', 'news_wsj_0157', 'jokes_jokes10', 'email_52999', 'wiki_lutheranism', 'journal_Article247_3500', 'letters_110CYL069', 'news_wsj_0032', 'wiki_historical_climatology', 'blog_detroit', 'news_NYTnewswire5'], ['wiki_dogs', 'email_lists-046-11489622', 'news_A1.E1-NEW', 'letters_119CWL041', 'fiction_A_Wasted_Day', 'wiki_judaism', 'technical_1468-6708-3-3', 'email_219123', 'news_wsj_0167', 'email_lists-003-2148080', 'wiki_narwahl', 'letters_116CUL033', 'email_175841', 'travel_WhereToHongKong', 'wiki_Major_depressive_disorder', 'letters_aspca1', 'blog_Effing-Idiot', 'email_lists-034-10069772', 'wiki_confederation', 'wiki_soccer', 'wiki_fiddle', 'email_52998', 'news_wsj_0006', 'jokes_jokes7', 'letters_112C-L012', 'essays_A_defense_of_Michael_Moore', 'news_NYTnewswire3', 'wiki_rummy'], ['email_210343', 'wiki_piano', 'letters_hsus4', 'news_wsj_0144', 'wiki_tennis', 'letters_110CYL200', 'email_lists-003-2137010', 'wiki_grand_coalition', 'news_wsj_0168', 'email_lists-003-2183485', 'wiki_political_party', 'technical_journal.pbio.0020001', 'email_lists-034-10073419', 'jokes_jokes6', 'wiki_surma_people', 'fiction_captured_moments', 'blog_Acephalous-Cant-believe', 'letters_114CUL060', 'blog_blog-monastery', 'email_50307', 'wiki_IslamGen', 'news_wsj_0135', 'essays_Bartok', 'email_lists-003-2133315', 'wiki_blackjack', 'letters_112C-L014', 'wiki_cats', 'news_NYTnewswire1'], ['email_175816', 'blog_lessig_blog-carbon', 'fiction_The_Black_Willow', 'wiki_empire', 'news_wsj_0026', 'letters_118CWL048', 'email_lists-003-2180740', 'letters_113CWL018', 'wiki_midwife', 'wiki_cows', 'email_54537', 'wiki_Cajon', 'letters_110CYL070', 'email_lists-034-10082707', 'email_52555', 'wiki_mandocello', 'letters_appalachian1', 'technical_1471-230X-2-21', 'news_wsj_0158', 'jokes_jokes12', 'wiki_huli_people', 'wiki_Henry_Burden', 'news_wsj_0068', 'email_23559', 'email_9159', 'wiki_trees', 'news_NYTnewswire4', 'blog_blog-jet-lag']])

    # learning curve fold:
    #fold_list = load_data(learning_curve = True, given_fold_list = [['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon', 'email_21257', 'letters_114CUL057', 'news_NYTnewswire8', 'wiki_bari_people', 'email_218920', 'letters_114CUL058', 'news_NYTnewswire9', 'wiki_baseball', 'email_219122', 'letters_114CUL059', 'news_wsj_0006', 'wiki_basketball', 'email_219123', 'letters_114CUL060', 'news_wsj_0026', 'wiki_blackjack', 'email_219257', 'letters_115CVL035', 'news_wsj_0027', 'wiki_bloods', 'email_221197', 'letters_115CVL036', 'news_wsj_0032', 'wiki_buddhismGen', 'email_230685', 'letters_115CVL037', 'news_wsj_0068', 'wiki_capitalism'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon', 'email_21257', 'letters_114CUL057', 'news_NYTnewswire8', 'wiki_bari_people', 'email_218920', 'letters_114CUL058', 'news_NYTnewswire9', 'wiki_baseball', 'email_219122', 'letters_114CUL059', 'news_wsj_0006', 'wiki_basketball', 'email_219123', 'letters_114CUL060', 'news_wsj_0026', 'wiki_blackjack', 'email_219257', 'letters_115CVL035', 'news_wsj_0027', 'wiki_bloods', 'email_221197', 'letters_115CVL036', 'news_wsj_0032', 'wiki_buddhismGen', 'email_230685', 'letters_115CVL037', 'news_wsj_0068', 'wiki_capitalism', 'email_234267', 'letters_116CUL032', 'news_wsj_0073', 'wiki_cats', 'email_234783', 'letters_116CUL033', 'news_wsj_0106', 'wiki_chemistry', 'email_23559', 'letters_116CUL034', 'news_wsj_0120', 'wiki_choir', 'email_49059', 'letters_117CWL008', 'news_wsj_0124', 'wiki_clavichord', 'email_50307', 'letters_117CWL009', 'news_wsj_0127', 'wiki_color_blindness', 'email_52201', 'letters_118CWL048', 'news_wsj_0132', 'wiki_confederation', 'email_52555', 'letters_118CWL049', 'news_wsj_0135', 'wiki_consumerlawsGen', 'email_52713'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon', 'email_21257', 'letters_114CUL057', 'news_NYTnewswire8', 'wiki_bari_people', 'email_218920', 'letters_114CUL058', 'news_NYTnewswire9', 'wiki_baseball', 'email_219122', 'letters_114CUL059', 'news_wsj_0006', 'wiki_basketball', 'email_219123', 'letters_114CUL060', 'news_wsj_0026', 'wiki_blackjack', 'email_219257', 'letters_115CVL035', 'news_wsj_0027', 'wiki_bloods', 'email_221197', 'letters_115CVL036', 'news_wsj_0032', 'wiki_buddhismGen', 'email_230685', 'letters_115CVL037', 'news_wsj_0068', 'wiki_capitalism', 'email_234267', 'letters_116CUL032', 'news_wsj_0073', 'wiki_cats', 'email_234783', 'letters_116CUL033', 'news_wsj_0106', 'wiki_chemistry', 'email_23559', 'letters_116CUL034', 'news_wsj_0120', 'wiki_choir', 'email_49059', 'letters_117CWL008', 'news_wsj_0124', 'wiki_clavichord', 'email_50307', 'letters_117CWL009', 'news_wsj_0127', 'wiki_color_blindness', 'email_52201', 'letters_118CWL048', 'news_wsj_0132', 'wiki_confederation', 'email_52555', 'letters_118CWL049', 'news_wsj_0135', 'wiki_consumerlawsGen', 'email_52713', 'letters_118CWL050', 'news_wsj_0136', 'wiki_cows', 'email_52998', 'letters_119CWL041', 'news_wsj_0144', 'wiki_democracyGen', 'email_52999', 'letters_602CZL285', 'news_wsj_0150', 'wiki_despotism', 'email_53536', 'letters_AMC2', 'news_wsj_0151', 'wiki_dictatorship', 'email_53555', 'letters_alumnifund1', 'news_wsj_0152', 'wiki_dogs', 'email_54261', 'letters_appalachian1', 'news_wsj_0157', 'wiki_empire', 'email_54262', 'letters_aspca1', 'news_wsj_0158', 'wiki_fiddle', 'email_54263'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon', 'email_21257', 'letters_114CUL057', 'news_NYTnewswire8', 'wiki_bari_people', 'email_218920', 'letters_114CUL058', 'news_NYTnewswire9', 'wiki_baseball', 'email_219122', 'letters_114CUL059', 'news_wsj_0006', 'wiki_basketball', 'email_219123', 'letters_114CUL060', 'news_wsj_0026', 'wiki_blackjack', 'email_219257', 'letters_115CVL035', 'news_wsj_0027', 'wiki_bloods', 'email_221197', 'letters_115CVL036', 'news_wsj_0032', 'wiki_buddhismGen', 'email_230685', 'letters_115CVL037', 'news_wsj_0068', 'wiki_capitalism', 'email_234267', 'letters_116CUL032', 'news_wsj_0073', 'wiki_cats', 'email_234783', 'letters_116CUL033', 'news_wsj_0106', 'wiki_chemistry', 'email_23559', 'letters_116CUL034', 'news_wsj_0120', 'wiki_choir', 'email_49059', 'letters_117CWL008', 'news_wsj_0124', 'wiki_clavichord', 'email_50307', 'letters_117CWL009', 'news_wsj_0127', 'wiki_color_blindness', 'email_52201', 'letters_118CWL048', 'news_wsj_0132', 'wiki_confederation', 'email_52555', 'letters_118CWL049', 'news_wsj_0135', 'wiki_consumerlawsGen', 'email_52713', 'letters_118CWL050', 'news_wsj_0136', 'wiki_cows', 'email_52998', 'letters_119CWL041', 'news_wsj_0144', 'wiki_democracyGen', 'email_52999', 'letters_602CZL285', 'news_wsj_0150', 'wiki_despotism', 'email_53536', 'letters_AMC2', 'news_wsj_0151', 'wiki_dictatorship', 'email_53555', 'letters_alumnifund1', 'news_wsj_0152', 'wiki_dogs', 'email_54261', 'letters_appalachian1', 'news_wsj_0157', 'wiki_empire', 'email_54262', 'letters_aspca1', 'news_wsj_0158', 'wiki_fiddle', 'email_54263', 'letters_att2', 'news_wsj_0159', 'wiki_free_will', 'email_54536', 'letters_audubon1', 'news_wsj_0160', 'wiki_go_fish', 'email_54537', 'letters_audubon2', 'news_wsj_0161', 'wiki_grand_coalition', 'email_8885', 'letters_defenders5', 'news_wsj_0165', 'wiki_gusli', 'email_9066', 'letters_guidedogs1', 'news_wsj_0167', 'wiki_hells_angels', 'email_9085', 'letters_hsus4', 'news_wsj_0168', 'wiki_herbalism', 'email_9159', 'wiki_hinduism', 'email_9191', 'wiki_historical_climatology', 'email_ENRON-pearson-email-25jul02', 'wiki_huli_people'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon', 'email_21257', 'letters_114CUL057', 'news_NYTnewswire8', 'wiki_bari_people', 'email_218920', 'letters_114CUL058', 'news_NYTnewswire9', 'wiki_baseball', 'email_219122', 'letters_114CUL059', 'news_wsj_0006', 'wiki_basketball', 'email_219123', 'letters_114CUL060', 'news_wsj_0026', 'wiki_blackjack', 'email_219257', 'letters_115CVL035', 'news_wsj_0027', 'wiki_bloods', 'email_221197', 'letters_115CVL036', 'news_wsj_0032', 'wiki_buddhismGen', 'email_230685', 'letters_115CVL037', 'news_wsj_0068', 'wiki_capitalism', 'email_234267', 'letters_116CUL032', 'news_wsj_0073', 'wiki_cats', 'email_234783', 'letters_116CUL033', 'news_wsj_0106', 'wiki_chemistry', 'email_23559', 'letters_116CUL034', 'news_wsj_0120', 'wiki_choir', 'email_49059', 'letters_117CWL008', 'news_wsj_0124', 'wiki_clavichord', 'email_50307', 'letters_117CWL009', 'news_wsj_0127', 'wiki_color_blindness', 'email_52201', 'letters_118CWL048', 'news_wsj_0132', 'wiki_confederation', 'email_52555', 'letters_118CWL049', 'news_wsj_0135', 'wiki_consumerlawsGen', 'email_52713', 'letters_118CWL050', 'news_wsj_0136', 'wiki_cows', 'email_52998', 'letters_119CWL041', 'news_wsj_0144', 'wiki_democracyGen', 'email_52999', 'letters_602CZL285', 'news_wsj_0150', 'wiki_despotism', 'email_53536', 'letters_AMC2', 'news_wsj_0151', 'wiki_dictatorship', 'email_53555', 'letters_alumnifund1', 'news_wsj_0152', 'wiki_dogs', 'email_54261', 'letters_appalachian1', 'news_wsj_0157', 'wiki_empire', 'email_54262', 'letters_aspca1', 'news_wsj_0158', 'wiki_fiddle', 'email_54263', 'letters_att2', 'news_wsj_0159', 'wiki_free_will', 'email_54536', 'letters_audubon1', 'news_wsj_0160', 'wiki_go_fish', 'email_54537', 'letters_audubon2', 'news_wsj_0161', 'wiki_grand_coalition', 'email_8885', 'letters_defenders5', 'news_wsj_0165', 'wiki_gusli', 'email_9066', 'letters_guidedogs1', 'news_wsj_0167', 'wiki_hells_angels', 'email_9085', 'letters_hsus4', 'news_wsj_0168', 'wiki_herbalism', 'email_9159', 'wiki_hinduism', 'email_9191', 'wiki_historical_climatology', 'email_ENRON-pearson-email-25jul02', 'wiki_huli_people', 'email_enron-thread-159550', 'wiki_human_rights', 'email_lists-003-2114716', 'wiki_judaism', 'email_lists-003-2121270', 'wiki_leafGen', 'email_lists-003-2125109', 'wiki_left-wing_politics', 'email_lists-003-2129640', 'wiki_lutheranism', 'email_lists-003-2133315', 'wiki_mandocello', 'email_lists-003-2137010', 'wiki_mantis_shrimp', 'email_lists-003-2144868', 'wiki_midwife', 'email_lists-003-2148080', 'wiki_narwahl', 'email_lists-003-2152883', 'wiki_neurology', 'email_lists-003-2171003', 'wiki_occultism', 'email_lists-003-2173878', 'wiki_okapi', 'email_lists-003-2180740', 'wiki_papyrus', 'email_lists-003-2183485', 'wiki_parapsychology'], ['blog_Acephalous-Cant-believe', 'email_12030', 'essays_A_defense_of_Michael_Moore', 'ficlets_1399', 'fiction_A_Wasted_Day', 'govt-docs_Env_Prot_Agency-nov1', 'jokes_jokes1', 'journal_Article247_327', 'letters_110CYL067', 'news_20000410_nyt-NEW', 'technical_1468-6708-3-1', 'travel_HistoryGreek', 'wiki_Alexander_Stepanovich_Popov', 'blog_Acephalous-Internet', 'email_12174', 'essays_Bartok', 'ficlets_1400', 'fiction_Nathans_Bylichka', 'govt-docs_chapter-10', 'jokes_jokes10', 'journal_Article247_328', 'letters_110CYL068', 'news_20000415_apw_eng-NEW', 'technical_1468-6708-3-3', 'travel_HistoryJerusalem', 'wiki_Archimedes_screw', 'blog_Anti-Terrorist', 'email_12176', 'essays_Black_and_white', 'ficlets_1401', 'fiction_The_Black_Willow', 'govt-docs_fcic_final_report_conclusions', 'jokes_jokes11', 'journal_Article247_3500', 'letters_110CYL069', 'news_20000419_apw_eng-NEW', 'technical_1471-2091-2-9', 'travel_HistoryLasVegas', 'wiki_Cajon', 'blog_Effing-Idiot', 'email_173252', 'essays_Homosexuality', 'ficlets_1402', 'fiction_captured_moments', 'jokes_jokes12', 'journal_VOL15_3', 'letters_110CYL070', 'news_20000424_nyt-NEW', 'technical_1471-213X-1-1', 'travel_IntroDublin', 'wiki_Harry_Coover', 'blog_Fermentation_Eminent-Domain', 'email_173906', 'essays_anth_essay_4', 'fiction_lw1', 'jokes_jokes2', 'letters_110CYL071', 'news_20020731-nyt', 'technical_1471-230X-2-21', 'travel_WhereToHongKong', 'wiki_Henry_Burden', 'blog_Fermentation_HR5034', 'email_174124', 'jokes_jokes3', 'letters_110CYL072', 'news_A1.E1-NEW', 'technical_journal.pbio.0020001', 'wiki_IslamGen', 'blog_How_soon-Fans', 'email_175448', 'jokes_jokes4', 'letters_110CYL200', 'news_A1.E2-NEW', 'wiki_Joseph_Swan', 'blog_How_soon-Lebron-James', 'email_175814', 'jokes_jokes5', 'letters_112C-L012', 'news_NYTnewswire1', 'wiki_Louis_Braille', 'blog_Italy', 'email_175816', 'jokes_jokes6', 'letters_112C-L013', 'news_NYTnewswire2', 'wiki_Major_depressive_disorder', 'blog_blog-jet-lag', 'email_175841', 'jokes_jokes7', 'letters_112C-L014', 'news_NYTnewswire3', 'wiki_Robert_Koch', 'blog_blog-monastery', 'email_176581', 'jokes_jokes8', 'letters_112C-L015', 'news_NYTnewswire4', 'wiki_american_football', 'blog_blog-new-years-resolutions', 'email_210343', 'letters_112C-L016', 'news_NYTnewswire5', 'wiki_ayoreo_people', 'blog_blog-varsity-athletics', 'email_211401', 'letters_113CWL017', 'news_NYTnewswire6', 'wiki_bagpipes', 'blog_detroit', 'email_211402', 'letters_113CWL018', 'news_NYTnewswire7', 'wiki_bandidos', 'blog_lessig_blog-carbon', 'email_21257', 'letters_114CUL057', 'news_NYTnewswire8', 'wiki_bari_people', 'email_218920', 'letters_114CUL058', 'news_NYTnewswire9', 'wiki_baseball', 'email_219122', 'letters_114CUL059', 'news_wsj_0006', 'wiki_basketball', 'email_219123', 'letters_114CUL060', 'news_wsj_0026', 'wiki_blackjack', 'email_219257', 'letters_115CVL035', 'news_wsj_0027', 'wiki_bloods', 'email_221197', 'letters_115CVL036', 'news_wsj_0032', 'wiki_buddhismGen', 'email_230685', 'letters_115CVL037', 'news_wsj_0068', 'wiki_capitalism', 'email_234267', 'letters_116CUL032', 'news_wsj_0073', 'wiki_cats', 'email_234783', 'letters_116CUL033', 'news_wsj_0106', 'wiki_chemistry', 'email_23559', 'letters_116CUL034', 'news_wsj_0120', 'wiki_choir', 'email_49059', 'letters_117CWL008', 'news_wsj_0124', 'wiki_clavichord', 'email_50307', 'letters_117CWL009', 'news_wsj_0127', 'wiki_color_blindness', 'email_52201', 'letters_118CWL048', 'news_wsj_0132', 'wiki_confederation', 'email_52555', 'letters_118CWL049', 'news_wsj_0135', 'wiki_consumerlawsGen', 'email_52713', 'letters_118CWL050', 'news_wsj_0136', 'wiki_cows', 'email_52998', 'letters_119CWL041', 'news_wsj_0144', 'wiki_democracyGen', 'email_52999', 'letters_602CZL285', 'news_wsj_0150', 'wiki_despotism', 'email_53536', 'letters_AMC2', 'news_wsj_0151', 'wiki_dictatorship', 'email_53555', 'letters_alumnifund1', 'news_wsj_0152', 'wiki_dogs', 'email_54261', 'letters_appalachian1', 'news_wsj_0157', 'wiki_empire', 'email_54262', 'letters_aspca1', 'news_wsj_0158', 'wiki_fiddle', 'email_54263', 'letters_att2', 'news_wsj_0159', 'wiki_free_will', 'email_54536', 'letters_audubon1', 'news_wsj_0160', 'wiki_go_fish', 'email_54537', 'letters_audubon2', 'news_wsj_0161', 'wiki_grand_coalition', 'email_8885', 'letters_defenders5', 'news_wsj_0165', 'wiki_gusli', 'email_9066', 'letters_guidedogs1', 'news_wsj_0167', 'wiki_hells_angels', 'email_9085', 'letters_hsus4', 'news_wsj_0168', 'wiki_herbalism', 'email_9159', 'wiki_hinduism', 'email_9191', 'wiki_historical_climatology', 'email_ENRON-pearson-email-25jul02', 'wiki_huli_people', 'email_enron-thread-159550', 'wiki_human_rights', 'email_lists-003-2114716', 'wiki_judaism', 'email_lists-003-2121270', 'wiki_leafGen', 'email_lists-003-2125109', 'wiki_left-wing_politics', 'email_lists-003-2129640', 'wiki_lutheranism', 'email_lists-003-2133315', 'wiki_mandocello', 'email_lists-003-2137010', 'wiki_mantis_shrimp', 'email_lists-003-2144868', 'wiki_midwife', 'email_lists-003-2148080', 'wiki_narwahl', 'email_lists-003-2152883', 'wiki_neurology', 'email_lists-003-2171003', 'wiki_occultism', 'email_lists-003-2173878', 'wiki_okapi', 'email_lists-003-2180740', 'wiki_papyrus', 'email_lists-003-2183485', 'wiki_parapsychology', 'email_lists-003-2205935', 'wiki_patolli', 'email_lists-034-10062451', 'wiki_philosophy_of_mind', 'email_lists-034-10066763', 'wiki_piano', 'email_lists-034-10069772', 'wiki_pintupi', 'email_lists-034-10073419', 'wiki_political_party', 'email_lists-034-10077455', 'wiki_rummy', 'email_lists-034-10082707', 'wiki_shoebill', 'email_lists-046-11485666', 'wiki_single-party_state', 'email_lists-046-11489622', 'wiki_sloth', 'wiki_soccer', 'wiki_stratocracy', 'wiki_surma_people', 'wiki_tennis', 'wiki_trees', 'wiki_venusflytrapGen', 'wiki_veterinary_medicine', 'wiki_volleyball', 'wiki_yodeling', 'wiki_yuqui_people', 'wiki_zebras']])

    for parameters in parameters_list:
        each_fold_macro_F1_list = []
        each_fold_result_list = []

        for i in range(len(fold_list)):
            print '----------------------------------------------------'
            print 'Training Fold ' + str(i+1) + '...'
            print '----------------------------------------------------'
            train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y = fold_list[i]
            word_embedding_dimension = test_X[0].size(-1)
            number_class = test_Y[0].size(-1)

            feed_data_cuda([train_X,train_Y,test_X])

            fold_best_result = None
            fold_best_model = None
            fold_best_macro = -1
            each_iteration_result_list = []
            each_iteration_best_model_list = []
            each_iteration_macro_Fscore_list = []

            for iteration in range(5):
                model = BaseSequenceLabeling(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                                             sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
                #model = BaseSequenceLabeling_LSTMEncoder(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                #                             sentence_zero_inithidden = parameters['sentence_zero_inithidden'], attention = None, num_layers = parameters['num_layers'], dropout = parameters['dropout'])

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
                if best_macro_Fscore > fold_best_macro:
                    fold_best_macro = best_macro_Fscore
                    fold_best_result = best_result
                    fold_best_model = best_model

            feed_data_cpu([train_X,train_Y,test_X])
            print '--------------------------------------------------------------------------'
            print 'Fold Best Result:'
            print 'Evaluate on all situation entity'
            print '-------------------------------------------------------------------------'
            print_evaluation_result(fold_best_result)

            fold_average_result = average_result(each_iteration_result_list)
            each_fold_result_list.append(fold_average_result)
            print '-------------------------------------------------------------------------'
            print 'Fold Average Result:'
            print 'Evaluate on all situation entity'
            print '-------------------------------------------------------------------------'
            each_fold_macro_F1_list.append(print_evaluation_result(fold_average_result)[0])

            print 'Fold Situation entity classification Macro F1_score std:' + str(np.std(each_iteration_macro_Fscore_list))       
            print 'Training Fold ' + str(i+1) + ' completed!'
            print 'Genre: ' + genre_list[i]
            print '-------------------------------------------------------------------------'    


        overall_average_result = average_result(each_fold_result_list)
        print '-------------------------------------------------------------------------'
        print 'Overall Average Result:'
        print 'Evaluate on all situation entity'
        print '-------------------------------------------------------------------------'
        print_evaluation_result(overall_average_result)
        print '-------------------------------------------------------------------------'
        print 'Overall Situation entity classification Macro F1_score std:' + str(np.std(each_fold_macro_F1_list))     

        print 'macro F1_score of each fold:  ' + str(each_fold_macro_F1_list) + '\n'
        print str(parameters)
        sys.stdout.flush()