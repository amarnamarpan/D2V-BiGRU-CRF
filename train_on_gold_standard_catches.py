import os
import sys
from wrap import Sequence
import numpy as np
import sentencify
import operator
import pickle as pic
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
config = tf.ConfigProto()

def get_tagged_words(catches, text):
	sents = sentencify.get_sents(text)
	x_arr = []
	y_arr = []
	for sent in sents:
		targets = ['O']*len(sent)
		lines=[]
		word_ind = 0
		max_word_ind = len(sent)
		while (word_ind < max_word_ind):
			for catch in catches:
				catch_len = len(catch.split(' '))
				if word_ind+catch_len <= max_word_ind:
					if (' '.join( sent[word_ind : word_ind+catch_len] )) == catch:
						catch_ind=0
						while(catch_ind < catch_len):
							if(catch_ind==0):
								targets[word_ind+catch_ind]='B-LEG'
							else:
								targets[word_ind+catch_ind]='I-LEG'
							catch_ind+=1
			word_ind+=1
		sent.append('.')
		targets.append('O')
		x_arr.append(sent)
		y_arr.append(targets)
	return x_arr, y_arr

def sanity_check(a,b):
	if len(a)!=len(b):
		return False

	for i,j in zip(a,b):
		if len(i)!=len(j):
			return False
	return True

from wrap import Sequence
from models import save_model

def train(docs_folder, catches_folder, models_folder, flist, d2v_modelname, catches_delimiter = ', '):
	x_train = []
	y_train = []
	doc_partition_indices = []
	prev = 0
	for fn in flist:
		try:
			text = open(os.path.join(docs_folder,fn),'r', encoding='iso-8859-1').read()
			ctcs = [v.lower() for v in open(os.path.join(catches_folder,fn), encoding='iso-8859-1').read().split(catches_delimiter)]
		except:
			continue
		x, y = get_tagged_words(ctcs, text)
		doc_partition_indices.append(len(x)+prev)
		prev += len(x)
		x_train.extend(x)
		y_train.extend(y)

	print(len(x_train),len(x_train[10]), len(y_train))
	if sanity_check(x_train,y_train)==False:
		raise ValueError('Sanity check failed')
	model= Sequence(word_embedding_dim=100,
	char_embedding_dim=25,
	word_lstm_size=100,
	char_lstm_size=25,
	d2v_modelname=d2v_modelname,
	doc_partition_indices=doc_partition_indices)
	if not os.path.exists(os.path.join(models_folder,"model_weights")):
		model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=20)
		model.save(os.path.join(models_folder,"model_weights"), os.path.join(models_folder,"model_params"), os.path.join(models_folder,"model_pre_proc"))
	else:
		print(os.path.join(models_folder,"model_weights")+' already exists.. ')

if __name__=='__main__':
	docs_folder = 'cleaned_cases/'
	delimitter_for_catchphrase_files = ', '
	catches_folder = 'catchwords/'
	models_folder = 'The_D2V_BiGRU_CRF_model/'
	d2v_modelname = 'D2Vmodel_for_insc_33545'
	if not os.path.exists(models_folder):
		os.mkdir(models_folder)
	files_list = os.listdir(docs_folder)
	train(docs_folder, catches_folder, models_folder, files_list, d2v_modelname, catches_delimiter=delimitter_for_catchphrase_files)
