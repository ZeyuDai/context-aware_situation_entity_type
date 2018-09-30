# -*- coding: utf-8 -*- 
import os 
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import re
import itertools
import cPickle
import copy
import numpy as np
from bs4 import BeautifulSoup

import torch
from torch.autograd import Variable

import gensim
import nltk
from nltk.tag import StanfordPOSTagger,StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import StanfordTokenizer

doc_list = []
test_doc_list = []
with open('./dataset/MASC_Wikipedia/train_test_split.csv','r') as f:
	for line in f.readlines():
		filename,genre,index,train_test =  line.split()
		filename = filename[:-4]
		if train_test == 'train' or train_test == 'test':
			doc_list.append(filename)
		if train_test == 'test':
			test_doc_list.append(filename)

# Load stored pos/ner parsing sentence
sentence_pos_ner_dict = {}
with open('../resource/masc_sentence_pos_ner_dict.pkl','r') as f:
	sentence_pos_ner_dict = cPickle.load(f)
	f.close()

connective_list = []
with open('../resource/explicit_connective.txt','r') as f:
	for line in f.readlines():
		connective_list.append(line.strip())
	connective_list = tuple(connective_list)
	f.close()

clause_count = 0
connective_count = 0
def sentence_startwith_connective(sentence):
	sentence = sentence.strip()
	if sentence[0] == '"':
		sentence = sentence[1:]
	if sentence.lower().startswith(connective_list):
		return True
	return False

def store_sentence_pos_ner_dict():
	with open('../resource/masc_sentence_pos_ner_dict.pkl','w') as f:
		cPickle.dump(sentence_pos_ner_dict,f)
		f.close()

# Load Google pretrained word2vec
model = gensim.models.Word2Vec.load_word2vec_format('../resource/GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.Word2Vec.load_word2vec_format('../resource/glove.840B.300d.w2vformat.txt', binary=False)

stanford_dir = '../resource/stanford-postagger-2016-10-31/'
modelfile = stanford_dir + 'models/english-left3words-distsim.tagger'
jarfile = stanford_dir + 'stanford-postagger.jar'
pos_tager = StanfordPOSTagger(modelfile, jarfile, encoding='utf8')
#print pos_tager.tag("Brack Obama lives in New York .".split())
st = StanfordTokenizer(jarfile, encoding='utf8')
#print st.tokenize('Among 33 men who worked closely with the substance, 28 have died -- more than three times the expected number. Four of the five surviving workers have asbestos-related diseases, including three with recently diagnosed cancer.')
stanford_dir = '../resource/stanford-ner-2016-10-31/'
modelfile = stanford_dir + 'classifiers/english.muc.7class.distsim.crf.ser.gz'
jarfile = stanford_dir + 'stanford-ner.jar'
ner_tager = StanfordNERTagger(modelfile, jarfile, encoding='utf8')
#print ner_tager.tag("In Jan. 5, Brack Obama lives in New York at 5:20 .".split())
#print ner_tager.tag(nltk.word_tokenize("Assets of the 400 taxable funds grew by $1.5 billion during the latest week, to $352.7 billion."))

vocab={}
def unknown_words(word,k=300):
	if word == '' or word in ['<SOS>','<EOS>']:
		return torch.zeros(k)
	if word not in vocab:
		vocab[word] = torch.rand(k)/2 - 0.25 
	return vocab[word] 

NER_LIST = ['ORGANIZATION','LOCATION','PERSON','MONEY','PERCENT','DATE','TIME']
PEN_TREEBANK_POS_LIST = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
def tansfer_word2vec(input_list,posner_flag=True,k=300):
	if posner_flag:
		pos_list,ner_list = input_list[0],input_list[1]
		embedding = torch.zeros(len(pos_list),k+len(PEN_TREEBANK_POS_LIST)+len(NER_LIST))

		for i in range(len(pos_list)):
			word,pos,ner = pos_list[i][0],pos_list[i][1],ner_list[i][1]

			if word in model:
				embedding[i,:k] = torch.from_numpy(model[word])
			else:
				embedding[i,:k] = unknown_words(word)

			if pos in PEN_TREEBANK_POS_LIST:
				embedding[i,k+PEN_TREEBANK_POS_LIST.index(pos)] = 1
			if ner in NER_LIST:
				embedding[i,k+len(PEN_TREEBANK_POS_LIST)+NER_LIST.index(ner)] = 1
		return embedding
	else:
		word_list = input_list
		embedding = torch.zeros(len(word_list),k)
		for i in range(len(word_list)):
			word = word_list[i]

			if word in model:
				embedding[i,:] = torch.from_numpy(model[word])
			else:
				embedding[i,:] = unknown_words(word)
		return embedding

def process_sentence(sentence, posner_flag = True, sentencemarker = False):
	if posner_flag:
		word_list = nltk.word_tokenize(sentence)
		if sentence not in sentence_pos_ner_dict:
			pos_list = pos_tager.tag(word_list)
			ner_list = ner_tager.tag(word_list)
			sentence_pos_ner_dict[sentence] = (copy.deepcopy(pos_list),copy.deepcopy(ner_list))
		else:
			pos_list = copy.deepcopy(sentence_pos_ner_dict[sentence][0])
			ner_list = copy.deepcopy(sentence_pos_ner_dict[sentence][1])
			assert len(pos_list) == len(word_list)

		if sentencemarker:
			pos_list.insert(0,('<SOS>',''))
			ner_list.insert(0,('<SOS>',''))

			pos_list.append(('<EOS>',''))
			ner_list.append(('<EOS>',''))

		return tansfer_word2vec((pos_list,ner_list), posner_flag = True)
	else:
		word_list = nltk.word_tokenize(sentence)
		#word_list = st.tokenize(sentence)
		
		if sentencemarker:
			word_list.insert(0,'<SOS>')
			word_list.append('<EOS>')

		return tansfer_word2vec(word_list, posner_flag = False)

entity_type_list = ['STATE','EVENT','REPORT','GENERIC_SENTENCE','GENERALIZING_SENTENCE','QUESTION','IMPERATIVE'] #'CANNOT_DECIDE'
def process_entity_type_label(entity_type):
	y = torch.zeros(len(entity_type_list))
	if entity_type in entity_type_list:
		y[entity_type_list.index(entity_type)] = 1
	return y

def transfer_docvec_labels(doc_clause_list, posner_flag = True, sentencemarker = False, connectivemarker = False):
	global connective_count
	global clause_count
	clause_embedding_list =[]
	connective_position_list = []
	y = torch.zeros(len(doc_clause_list),len(entity_type_list))
	eos_position_list = []
	doc_length = 0

	for i in range(len(doc_clause_list)):
		clause_text,entity_type = doc_clause_list[i][0],doc_clause_list[i][1]
		y[i,:] = process_entity_type_label(entity_type)

		clause_count += 1
		if connectivemarker:
			if sentence_startwith_connective(clause_text):
				connective_count += 1
				if clause_text.strip()[0] == '"':
					connective_position_list.append(doc_length+1)
				else:
					connective_position_list.append(doc_length)
			else:
				connective_position_list.append(-1)

		clause_embedding = process_sentence(clause_text, posner_flag = posner_flag, sentencemarker = sentencemarker)
		clause_embedding_list.append(clause_embedding)
		doc_length = doc_length + clause_embedding.size(0)
		eos_position_list.append(doc_length)

	doc_embedding = torch.cat(clause_embedding_list)
	doc_embedding = doc_embedding.view(1,-1,doc_embedding.size(-1))

	return doc_embedding,eos_position_list,connective_position_list,y

def process_doc(doc_path):
	doc = open(doc_path,'r')
	xml = BeautifulSoup(doc,"html.parser")

	clause_list = []
	for clause in xml.find_all('segment'):
		end = int(clause.attrs['end'])
		clause_text = unicode(clause.find('text').string)
		label = 'CANNOT_DECIDE'
		annotation = clause.find('annotation', attrs={"annotator":"gold"})
		if annotation.has_attr('setype'):
			label = annotation.attrs['setype']

		clause_list.append((clause_text,label,end))
	return clause_list

def process_paragraph(doc_clause_list,raw_doc_path):
	raw_doc = open(raw_doc_path,'r')
	raw_para_boundary_list = [m.start() + 1 for m in re.finditer('\n\n', raw_doc.read())]
	raw_para_boundary_list += [float('inf')]
	paras_clause_list = []

	index = 0
	for raw_para_boundary in raw_para_boundary_list:
		para_clause_list = []

		while index < len(doc_clause_list):
			end_index = doc_clause_list[index][2]
			if end_index <= raw_para_boundary:
				para_clause_list.append(doc_clause_list[index])
				index += 1
			else:
				break

		if len(para_clause_list) > 0:
			paras_clause_list.append(para_clause_list)

	return paras_clause_list

para_length_list = []
def process_fold(fold_doc_list):
	global para_length_list
	doc_folder_path = './dataset/MASC_Wikipedia/annotations_xml/'
	raw_doc_folder_path = './dataset/MASC_Wikipedia/raw_text/'

	print "total number of documents:" + str(len(fold_doc_list))
	y_total = torch.zeros(len(entity_type_list))
	doc_dict = {}

	#para_text_lists = []
	#para_y_lists = []

	for i in range(len(fold_doc_list)):
		if i % 10 == 0:
			print i

		fold_doc_path = os.path.join(doc_folder_path,fold_doc_list[i]+ '.xml')
		fold_raw_doc_path = os.path.join(raw_doc_folder_path,fold_doc_list[i]+ '.txt')
		doc_clause_list =  process_doc(fold_doc_path)
		paras_clause_list = process_paragraph(doc_clause_list,fold_raw_doc_path)

		doc_embedding_list = []
		doc_label_length_list = []
		eos_position_lists = []
		connective_position_lists = []
		y_list = []

		'''clause_paralength_list = []
		for para_clause_list in paras_clause_list:
			for clause in para_clause_list:
				_,_,_,y = transfer_docvec_labels([clause], posner_flag = True)
				if torch.sum(y) == 0:
					continue

				clause_paralength_list.append(len(para_clause_list))

		print clause_paralength_list
		doc_dict[fold_doc_list[i]] = clause_paralength_list

	with open('data/masc_paragraph_dictformat_clause_paralength_list.pkl','w+') as f:
		cPickle.dump(doc_dict,f)
		f.close()'''

		for para_clause_list in paras_clause_list:
			doc_embedding,eos_position_list,connective_position_list,y = transfer_docvec_labels(para_clause_list, posner_flag = True, connectivemarker = True)
			if torch.sum(y) == 0:
				continue

			para_length_list.append(len(para_clause_list))

			doc_label_length_list.append(int(torch.sum(y)))
			y_total = y_total + torch.sum(y,0)

			#para_text_lists.append(para_clause_list)
			#para_y_lists.append(y)
			print para_clause_list
			print y

			doc_embedding = Variable(doc_embedding, requires_grad = False)
			y = Variable(y, requires_grad = False)

			doc_embedding_list.append(doc_embedding)
			eos_position_lists.append(eos_position_list)
			connective_position_lists.append(connective_position_list)
			y_list.append(y)

		if len(connective_position_list) != 0:
			doc_dict[fold_doc_list[i]] = [(doc_embedding_list,doc_label_length_list,eos_position_lists,connective_position_lists),y_list]
		else:
			doc_dict[fold_doc_list[i]] = [(doc_embedding_list,doc_label_length_list,eos_position_lists),y_list]

	'''with open('./data/masc_paragraph_testdata.pkl','w+') as f:
		cPickle.dump([para_text_lists,para_y_lists],f)
		f.close()'''

	print 'entity type distribution'
	print entity_type_list
	print y_total
	return doc_dict


#process_fold(test_doc_list)
masc_data = process_fold(doc_list)

print 'total number of para:' +  str(len(para_length_list))
print 'average para length: ' + str(sum(para_length_list) / float(len(para_length_list)))
print 'para length distribution: ' + str(np.unique(para_length_list, return_counts=True))
print 'connective count: ' + str(connective_count)
print 'connective percentage: ' + str(connective_count*1.0 / clause_count)
store_sentence_pos_ner_dict()

#with open('data/masc_paragraph_addposnerembedding_dictformat.pt','w+') as outfile:
#	torch.save(masc_data,outfile)