import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from CRF_layer import CRF

use_cuda = torch.cuda.is_available()

class SoftAttention(nn.Module):
	def __init__(self, input_size, hidden_size = 0, attention_function = 'feedforward', nonlinear = True, temporal= False):
		"""
		soft attention followed the Luong's https://arxiv.org/abs/1508.04025
		input_size: the dimension of each hidden states
		hidden_size: only used when attention_function == 'feedforward' or 'concat', decide the dimension of v, W, W1 and W2 (use input_size as default)
		attention_function:  decide how to calculate the weight of each encoder hidden states
			si: the ith decoder hidden states
			hj: the jth encoder hidden states

			eij = 	transpose(si) * hj       when attention_function == 'dot'
			eij = 	transpose(si) * W * hj   when attention_function == 'general'
			eij =   transpose(v) * tanh(W*[si;hj])	   when attention_function == 'concat'
			eij =   transpose(v) * tanh(W1*si + W2*hj)  when attention_function == 'feedforward'

			aij = softmax(eij)
			ci = sum(aij*hj)

		nolinear: when True, si_tilde = tanh(W[ci;si]) 
				  when False, si_tilde = ci
		"""
		super(SoftAttention, self).__init__()
		if hidden_size <=0:
			hidden_size = input_size

		self.attention_function = attention_function
		self.input_size = input_size
		self.nonlinear = nonlinear
		self.temporal = temporal

		if attention_function == 'feedforward':
			self.linear_W1 = nn.Linear(input_size, hidden_size, bias=False)
			self.linear_W2 = nn.Linear(input_size, hidden_size, bias=False)
			self.linear_v = nn.Linear(hidden_size, 1, bias=False)
		elif attention_function == 'general':
			self.linear_W = nn.Linear(input_size, input_size, bias=False)
		elif attention_function == 'concat':
			self.linear_W = nn.Linear(input_size*2, hidden_size*2, bias=False)
			self.linear_v = nn.Linear(hidden_size*2, 1, bias=False)

		self.softmax = nn.Softmax()
		self.tanh = nn.Tanh()

		if nonlinear:
			self.linear_out = nn.Linear(input_size*2, input_size, bias=False)
		if temporal:
			self.attn_history = []

	def forward(self, input, context):
		"""
		input:   si  batch x input_size
		context: h   batch x sourceL x input_size
		"""
		si = input
		if self.attention_function == 'feedforward':
			si = self.linear_W1(si)			# batch x hidden_size

			h = context.view(-1,self.input_size) # (batch*sourceL) x input_size
			h = self.linear_W2(h)			# (batch*sourceL) x hidden_size

			attn = self.linear_v(self.tanh(h + torch.cat([si]*context.size(1),0))) # (batch*sourceL) x 1
			attn = attn.view(context.size(0),context.size(1)) # batch x sourceL

		elif self.attention_function == 'dot':
			si = si.view(si.size(0),si.size(1),1) # batch x input_size x 1
			attn = torch.bmm(context, si).squeeze(2) # batch x sourceL

		elif self.attention_function == 'general':
			si = self.linear_W(si)			# batch x input_size
			si = si.view(si.size(0),si.size(1),1) # batch x input_size x 1
			attn = torch.bmm(context, si).squeeze(2) # batch x sourceL

		elif self.attention_function == 'concat':
			h = context.view(-1,self.input_size) # (batch*sourceL) x input_size

			attn = self.linear_v(self.tanh(self.linear_W(torch.cat([torch.cat([si]*context.size(1),0),h], dim=-1)))) # (batch*sourceL) x 1
			attn = attn.view(context.size(0),context.size(1)) # batch x sourceL

		if self.temporal:
			if len(self.attn_history) > 1:
				attn_sum = torch.sum(torch.cat(self.attn_history[:-1],0),0) # batch x sourceL
				self.attn_history.append(attn)
				attn = attn / attn_sum
			else:
				self.attn_history.append(attn)

		attn = self.softmax(attn)  # batch x sourceL
		attn3 = attn.view(attn.size(0), 1, attn.size(1)) # batch x 1 x sourceL
		ci = torch.bmm(attn3,context).squeeze(1) #contect vecor:  batch x input_size

		si_tilde = ci
		if self.nonlinear:
			si_tilde = self.tanh(self.linear_out(torch.cat((ci, input), dim=-1))) #batch x input_size

		return si_tilde, attn

	def clean_history(self):
		self.attn_history = []

'''
two word-level & clause-level LSTM encoder
input_size: word embedding dimension
hidden_size: output dimension of lstm (when hidden_size <= 0, use input_size as default)
sentence_embedding_type: ['last','mean','sum','max'] 
 						decide how to calculate clause embedding representation vector from a list of words representation vectors
sentence_zero_inithidden: whether to reset initial word_hidden to zero for each clause's word_level_lstm calculation	
'''
class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size = 0, batch_size = 1, num_layers = 1, dropout = 0, bidirectional = True, batch_first = True, sentence_embedding_type = 'last', sentence_zero_inithidden = False):
		super(Encoder, self).__init__()

		self.batch_size = batch_size
		self.input_dimension = input_size

		if hidden_size <= 0:
			hidden_size = self.input_dimension
		if hidden_size % 2 != 0 and bidirectional:
			hidden_size = hidden_size - 1

		self.hidden_size = hidden_size
		self.dropout = dropout
		self.batch_first = batch_first
		self.bidirectional = bidirectional
		self.sentence_embedding_type = sentence_embedding_type
		self.sentence_zero_inithidden = sentence_zero_inithidden

		if type(num_layers) == type(1):
			self.word_level_num_layers = num_layers
			self.sentence_level_num_layers = num_layers
		else:
			assert len(num_layers) == 2
			self.word_level_num_layers = num_layers[0]
			self.sentence_level_num_layers = num_layers[1]

		if self.bidirectional:
			self.lstm_unit = self.hidden_size // 2
		else:
			self.lstm_unit = self.hidden_size

		self.word_level_lstm = nn.LSTM(
			self.input_dimension,
			self.lstm_unit,
			self.word_level_num_layers,
			bidirectional=self.bidirectional,
			batch_first=self.batch_first,
			dropout=self.dropout
		)

		self.sentence_level_lstm = nn.LSTM(
			self.hidden_size,
			self.lstm_unit,
			self.sentence_level_num_layers,
			bidirectional=self.bidirectional,
			batch_first=self.batch_first,
			dropout=self.dropout
		)
		self.Dropout = nn.Dropout(dropout)

	def forward(self,input,eos_position_list,connective_position_list=None):
		word_hidden,sentence_hidden = self.initHidden()
		input = self.Dropout(input)

		if self.batch_size == 1:
			#process sample one by one
			if self.sentence_zero_inithidden:
				sentence_output_list = []
				prev_eos_position = 0
				for eos_position in eos_position_list:
					word_hidden,_ = self.initHidden()
					sentence_input = input[:,prev_eos_position:eos_position,:]
					sentence_output,_ = self.word_level_lstm(sentence_input,word_hidden)

					prev_eos_position = eos_position
					sentence_output_list.append(sentence_output)

				word_level_output = torch.cat(sentence_output_list, dim = 1)
			else:
				word_level_output,word_hidden = self.word_level_lstm(input, word_hidden)

			sentence_embedding_list = []
			prev_eos_position = 0
			for i in range(len(eos_position_list)):
				eos_position = eos_position_list[i]
				connective_position = -1
				if connective_position_list is not None:
					connective_position = connective_position_list[i]

				if connective_position != -1:
					#sentence_embedding = word_level_output[0,connective_position,:]
					if connective_position+1 < eos_position:
						sentence_embedding = torch.max(word_level_output[0,connective_position+1:eos_position,:],0)[0]
					else:
						sentence_embedding = word_level_output[0,connective_position,:]
				elif self.sentence_embedding_type == 'last':
					sentence_embedding = word_level_output[0,eos_position-1,:]
				elif self.sentence_embedding_type == 'sum':
					sentence_embedding = torch.sum(word_level_output[0,prev_eos_position:eos_position,:],0)
				elif self.sentence_embedding_type == 'mean':
					sentence_embedding = torch.mean(word_level_output[0,prev_eos_position:eos_position,:],0)
				elif self.sentence_embedding_type == 'max':
					sentence_embedding = torch.max(word_level_output[0,prev_eos_position:eos_position,:],0)[0]
				elif self.sentence_embedding_type == 'self-attentive':
					# apply self-attentive sentence embedding in https://arxiv.org/abs/1703.03130
					print 'To do later!'
					sys.exit()

				prev_eos_position = eos_position
				sentence_embedding_list.append(sentence_embedding)

			sentence_level_input = torch.stack(sentence_embedding_list)
			sentence_level_input = sentence_level_input.view(1,-1,self.hidden_size)

			sentence_level_input = self.Dropout(sentence_level_input)
			sentence_level_output,sentence_hidden = self.sentence_level_lstm(sentence_level_input, sentence_hidden)
			#sentence_level_output = self.Dropout(sentence_level_output)

			if self.sentence_zero_inithidden:
				word_hidden = sentence_hidden

			return word_level_output,sentence_level_output,word_hidden,sentence_hidden
		else:
			# To do later: process a batch of samples
			print "Don't support larger batch size now!"
			sys.exit()

	def initHidden(self):
		if self.bidirectional:
			h0_word = Variable(torch.zeros(
				2*self.word_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

			c0_word = Variable(torch.zeros(
				2*self.word_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

			h0_sentence = Variable(torch.zeros(
				2*self.sentence_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

			c0_sentence = Variable(torch.zeros(
				2*self.sentence_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

		else:
			h0_word = Variable(torch.zeros(
				self.word_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

			c0_word = Variable(torch.zeros(
				self.word_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

			h0_sentence = Variable(torch.zeros(
				self.sentence_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

			c0_sentence = Variable(torch.zeros(
				self.sentence_level_num_layers,
				self.batch_size,
				self.lstm_unit
			))

		h0_word = h0_word.cuda() if use_cuda else h0_word
		c0_word = c0_word.cuda() if use_cuda else c0_word
		h0_sentence = h0_sentence.cuda() if use_cuda else h0_sentence
		c0_sentence = c0_sentence.cuda() if use_cuda else c0_sentence

		return (h0_word,c0_word),(h0_sentence,c0_sentence)



'''
one word-level LSTM encoder
input_size: word embedding dimension
hidden_size: output dimension of lstm (when hidden_size <= 0, use input_size as default)
sentence_embedding_type: ['last','mean','sum','max'] 
 						decide how to calculate clause embedding representation vector from a list of words representation vectors
sentence_zero_inithidden: whether to reset initial word_hidden to zero for each clause's word_level_lstm calculation	
'''
class LSTMEncoder(nn.Module):
	def __init__(self, input_size, hidden_size = 0, batch_size = 1, num_layers = 1, dropout = 0, bidirectional = True, batch_first = True, sentence_embedding_type = 'last', sentence_zero_inithidden = False):
		super(LSTMEncoder, self).__init__()

		self.batch_size = batch_size
		self.input_dimension = input_size

		if hidden_size <= 0:
			hidden_size = self.input_dimension
		if hidden_size % 2 != 0 and bidirectional:
			hidden_size = hidden_size - 1

		self.hidden_size = hidden_size
		self.dropout = dropout
		self.batch_first = batch_first
		self.bidirectional = bidirectional
		self.sentence_embedding_type = sentence_embedding_type
		self.sentence_zero_inithidden = sentence_zero_inithidden

		self.num_layers = num_layers

		if self.bidirectional:
			self.lstm_unit = self.hidden_size // 2
		else:
			self.lstm_unit = self.hidden_size

		self.word_level_lstm = nn.LSTM(
			self.input_dimension,
			self.lstm_unit,
			self.num_layers,
			bidirectional=self.bidirectional,
			batch_first=self.batch_first,
			dropout=self.dropout
		)

		self.Dropout = nn.Dropout(dropout)

	def forward(self,input,eos_position_list,connective_position_list=None):
		word_hidden = self.initHidden()
		input = self.Dropout(input)

		if self.batch_size == 1:
			#process sample one by one
			if self.sentence_zero_inithidden:
				sentence_output_list = []
				prev_eos_position = 0
				for eos_position in eos_position_list:
					word_hidden = self.initHidden()
					sentence_input = input[:,prev_eos_position:eos_position,:]
					sentence_output,_ = self.word_level_lstm(sentence_input,word_hidden)

					prev_eos_position = eos_position
					sentence_output_list.append(sentence_output)

				word_level_output = torch.cat(sentence_output_list, dim = 1)
			else:
				word_level_output,word_hidden = self.word_level_lstm(input, word_hidden)

			sentence_embedding_list = []
			prev_eos_position = 0
			for i in range(len(eos_position_list)):
				eos_position = eos_position_list[i]
				connective_position = -1
				if connective_position_list is not None:
					connective_position = connective_position_list[i]

				if connective_position != -1:
					sentence_embedding = word_level_output[0,connective_position,:]
				elif self.sentence_embedding_type == 'last':
					sentence_embedding = word_level_output[0,eos_position-1,:]
				elif self.sentence_embedding_type == 'sum':
					sentence_embedding = torch.sum(word_level_output[0,prev_eos_position:eos_position,:],0)
				elif self.sentence_embedding_type == 'mean':
					sentence_embedding = torch.mean(word_level_output[0,prev_eos_position:eos_position,:],0)
				elif self.sentence_embedding_type == 'max':
					sentence_embedding = torch.max(word_level_output[0,prev_eos_position:eos_position,:],0)[0]
				elif self.sentence_embedding_type == 'self-attentive':
					# apply self-attentive sentence embedding in https://arxiv.org/abs/1703.03130
					print 'To do later!'
					sys.exit()

				prev_eos_position = eos_position
				sentence_embedding_list.append(sentence_embedding)

			sentence_level_output = torch.stack(sentence_embedding_list)
			sentence_level_output = sentence_level_output.view(1,-1,self.hidden_size)
			#sentence_level_output = self.Dropout(sentence_level_output)

			return word_level_output, sentence_level_output, word_hidden, word_hidden
		else:
			#To do later: process a batch of samples
			print "Don't support larger batch size now!"
			sys.exit()

	def initHidden(self):
		if self.bidirectional:
			h0_word = Variable(torch.zeros(
				2*self.num_layers,
				self.batch_size,
				self.lstm_unit
			))

			c0_word = Variable(torch.zeros(
				2*self.num_layers,
				self.batch_size,
				self.lstm_unit
			))
		else:
			h0_word = Variable(torch.zeros(
				self.num_layers,
				self.batch_size,
				self.lstm_unit
			))

			c0_word = Variable(torch.zeros(
				self.num_layers,
				self.batch_size,
				self.lstm_unit
			))

		h0_word = h0_word.cuda() if use_cuda else h0_word
		c0_word = c0_word.cuda() if use_cuda else c0_word

		return (h0_word,c0_word)



class BaseSequenceLabeling(nn.Module):
	def __init__(self, input_size, output_size, hidden_size = 0, sentence_embedding_type = 'last', sentence_zero_inithidden = False, attention = None, batch_size = 1, num_layers = 1, dropout = 0, bidirectional = True, batch_first = True):
		super(BaseSequenceLabeling, self).__init__()

		if hidden_size <= 0:
			hidden_size = input_size
		if hidden_size % 2 != 0 and bidirectional:
			hidden_size = hidden_size - 1
		self.batch_size = batch_size
		self.attention = attention

		self.encoder = Encoder(input_size,hidden_size, batch_size=batch_size, num_layers=num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = batch_first, sentence_embedding_type = sentence_embedding_type, sentence_zero_inithidden = sentence_zero_inithidden)
		if self.attention:
			self.soft_attention = SoftAttention(hidden_size, attention_function = self.attention, nonlinear = False, temporal = False)

		#self.connective_out = nn.Linear(hidden_size*2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

		self.softmax = nn.LogSoftmax()
		self.Dropout = nn.Dropout(dropout)

	def forward(self, input, eos_position_list, connective_position_list = None):
		word_level_output,sentence_level_output,_,_ = self.encoder(input,eos_position_list,connective_position_list)

		if self.batch_size == 1:
			output_list = []
			prev_eos = 0
			#process sample one by one			
			for i in range(len(eos_position_list)):
				if self.attention:
					output,_ = self.soft_attention(sentence_level_output[:,i,:],word_level_output[:,prev_eos:eos_position_list[i],:])
				else:
					output = sentence_level_output[:,i,:].view(1,-1)
				prev_eos = eos_position_list[i]

				'''# contain a connective:
				if connective_position_list is not None and connective_position_list[i] != -1:
					connective_embedding = word_level_output[:,connective_position_list[i],:].view(1,-1)
					output = self.connective_out(torch.cat([output,connective_embedding],dim = -1))

					if i >= 1 and connective_position_list[i-1] == -1:
						output_list[-1] = self.connective_out(torch.cat([output_list[-1],connective_embedding],dim = -1))'''

				output = self.Dropout(output)
				output_list.append(output)

			return self.softmax(self.out(torch.cat(output_list)))
		else:
			# To do later: process a batch of samples
			print "Don't support larger batch size now!"
			sys.exit()



class BaseSequenceLabeling_LSTMEncoder(nn.Module):
	def __init__(self, input_size, output_size, hidden_size = 0, sentence_embedding_type = 'last', sentence_zero_inithidden = False, attention = None, batch_size = 1, num_layers = 1, dropout = 0, bidirectional = True, batch_first = True):
		super(BaseSequenceLabeling_LSTMEncoder, self).__init__()

		if hidden_size <= 0:
			hidden_size = input_size
		if hidden_size % 2 != 0 and bidirectional:
			hidden_size = hidden_size - 1
		self.batch_size = batch_size
		self.attention = attention

		self.encoder = LSTMEncoder(input_size,hidden_size, batch_size=batch_size, num_layers=num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = batch_first, sentence_embedding_type = sentence_embedding_type, sentence_zero_inithidden = sentence_zero_inithidden)
		if self.attention:
			self.soft_attention = SoftAttention(hidden_size, attention_function = self.attention, nonlinear = False, temporal = False)

		self.out = nn.Linear(hidden_size, output_size)

		self.softmax = nn.LogSoftmax()
		self.Dropout = nn.Dropout(dropout)

	def forward(self, input, eos_position_list, connective_position_list = None):
		word_level_output,sentence_level_output,_,_ = self.encoder(input,eos_position_list,connective_position_list)

		if self.batch_size == 1:
			output_list = []
			prev_eos = 0
			#process sample one by one			
			for i in range(len(eos_position_list)):
				if self.attention:
					output,_ = self.soft_attention(sentence_level_output[:,i,:],word_level_output[:,prev_eos:eos_position_list[i],:])
				else:
					output = sentence_level_output[:,i,:].view(1,-1)
				prev_eos = eos_position_list[i]

				output = self.Dropout(output)
				output_list.append(output)

			return self.softmax(self.out(torch.cat(output_list)))
		else:
			# To do later: process a batch of samples
			print "Don't support larger batch size now!"
			sys.exit()



class BiLSTMCRF(nn.Module):
	def __init__(self, input_size, output_size, hidden_size = 0, sentence_embedding_type = 'last', sentence_zero_inithidden = False, attention = None, crf_decode_method = 'viterbi', loss_function  = 'likelihood', batch_size = 1, num_layers = 1, dropout = 0, bidirectional = True, batch_first = True):
		super(BiLSTMCRF, self).__init__()

		if hidden_size <= 0:
			hidden_size = input_size
		if hidden_size % 2 != 0 and bidirectional:
			hidden_size = hidden_size - 1
		self.batch_size = batch_size
		self.attention = attention
		self.crf_decode_method = crf_decode_method
		self.loss_function = loss_function

		self.encoder = Encoder(input_size,hidden_size, batch_size=batch_size, num_layers=num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = batch_first, sentence_embedding_type = sentence_embedding_type, sentence_zero_inithidden = sentence_zero_inithidden)
		if self.attention:
			self.soft_attention = SoftAttention(hidden_size, attention_function = self.attention, nonlinear = False, temporal = False)
		self.out = nn.Linear(hidden_size, output_size)

		self.Dropout = nn.Dropout(dropout)
		self.CRF = CRF(output_size)
		#self.softmax = nn.Softmax()

	def _get_lstm_features(self, input, eos_position_list, crf_target, connective_position_list = None):
		_,sentence_level_output,_,_ = self.encoder(input,eos_position_list,connective_position_list)

		if self.batch_size == 1:
			#process sample one by one
			output_list = []
			prev_eos = 0
			target_seq = []

			for i in range(len(eos_position_list)):
				if self.attention:
					output,_ = self.soft_attention(sentence_level_output[:,i,:],word_level_output[:,prev_eos:eos_position_list[i],:])
				else:
					output = sentence_level_output[:,i,:].view(1,-1)
				prev_eos = eos_position_list[i]

				if crf_target[i] >=0:
					output = self.Dropout(output)
					output = self.out(output)
					output_list.append(output)
					target_seq.append(crf_target[i])
		else:
			# To do later: process a batch of samples
			print "Don't support larger batch size now!"
			sys.exit()

		return torch.cat(output_list), torch.LongTensor(target_seq)

	def get_loss(self, input, eos_position_list, target, connective_position_list = None):
		# Get the emission scores from the BiLSTM
		feats,target_seq = self._get_lstm_features(input, eos_position_list, self.prepare_sequence(target),connective_position_list)

 		if self.loss_function == 'likelihood':
			return self.CRF._get_neg_log_likilihood_loss(feats, target_seq)

	def forward(self, input, eos_position_list, target, connective_position_list = None):
		# Get the emission scores from the BiLSTM
		feats,_ = self._get_lstm_features(input, eos_position_list, self.prepare_sequence(target),connective_position_list)

		# Find the best path, given the features.
		if self.crf_decode_method == 'marginal':
			score, tag_seq = self.CRF._marginal_decode(feats)
		elif self.crf_decode_method == 'viterbi':
			score, tag_seq = self.CRF._viterbi_decode(feats)

		target = target.abs()
		predict = torch.zeros(target.size())
		j = 0
		for i in range(predict.size(0)):
			if torch.max(target[i,:]).data[0] > 0:
				predict[i,tag_seq[j]] = 1
				j += 1

		return Variable(predict)

	def prepare_sequence(self,target):
		target = target.abs()
		max_value,indexs = torch.max(target,1)
		tensor = []

		for i in range(indexs.size(0)):
			if max_value[i].data[0] > 0:
				tensor.append(indexs[i].data[0])
			else:
				tensor.append(-1)
		return tensor
