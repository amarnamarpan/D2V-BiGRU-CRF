import os
import sys
from wrap import Sequence
import numpy as np
import sentencify
import operator
import pickle as pic
from gensim.models import doc2vec


def get_X_test(text):
	sents = sentencify.get_sents(text)
	x_arr = []
	for sent in sents:
		sent.append('.')
		x_arr.append(sent)
	return x_arr


def get_catches_from_tagged_words(words, tags, scores):
	if len(words) != len(tags):
		raise ValueError('The length of *words* and *tags* do not match..')
	begun=False
	t_catch=''
	catch_lst=[]
	catch_dict={}
	score=0
	n_words=0
	for i in range(len(words)):
		word = words[i]
		tag = tags[i]
		prev_score = score
		score = scores[i]
		if tag=='B-LEG':
			if begun==True:
				catch_lst.append(t_catch.strip())
				score = (prev_score+score)/n_words
				if t_catch in catch_dict:
					catch_dict[t_catch].append(score)
				else:
					catch_dict[t_catch]=[score]
				score = 0
			t_catch=word
			begun=True
			n_words=1
		elif tag=='I-LEG':
			t_catch=t_catch+' '+word
			n_words+=1
			score+=prev_score
		else:	#tag == 'o'
			if begun==True:
				catch_lst.append(t_catch.strip())
				score = (prev_score+score)/n_words
				if t_catch in catch_dict:
					catch_dict[t_catch].append(score)
				else:
					catch_dict[t_catch]=[score]
			begun=False
			t_catch=''
			score=0
			n_words=0

	for each in catch_dict.keys():
		catch_dict[each] = sum(catch_dict[each]) / len(catch_dict[each])
	c_dict = sorted(catch_dict.items(), key=operator.itemgetter(1))
	scores = [i[1] for i in c_dict]
	words = [i[0] for i in c_dict]
	for i,w in enumerate(words):
		catch_dict[w] = scores[i]
	words.reverse()
	scores.reverse()
	return words

labels = ['X','O','B-LEG','I-LEG']

def get_annotations(docs_folder, out_folder, models_folder, d2v_modelname):
	model = Sequence()
	model = model.load(os.path.join(models_folder,"model_weights"), os.path.join(models_folder,"model_params"), os.path.join(models_folder,"model_pre_proc"))
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)
	flist = os.listdir(docs_folder)
	d2v_model = doc2vec.Doc2Vec.load(d2v_modelname)
	d2v_dim = len(d2v_model.infer_vector(['the','god'],steps=1))
	for fn in flist:
		try:
			text = open(docs_folder+fn,'r', encoding='iso-8859-1').read()
		except:
			continue
		x_test = get_X_test(text)
		d2v_vectors = np.zeros((len(x_test), d2v_dim)).astype('float64')
		tokenized_doc = []
		for sent in x_test:
			tokenized_doc.extend(sent)
		d2v_vec = d2v_model.infer_vector(tokenized_doc,steps=15)
		for i,sent in enumerate(x_test):
			d2v_vectors[i] = d2v_vec
		probs = model.predict(x_test,d2v_vectors)
		pred_tags = probs.argmax(2)
		words = []
		for i,sent in enumerate(x_test):
			cur_annotations = get_catches_from_tagged_words(x_test[i], [labels[int(fg)%len(labels)] for fg in pred_tags[i][:len(x_test[i])]], [1]*len(x_test[i]))
			words.extend(cur_annotations)
		words = list(set(words))
		open(os.path.join(out_folder,(fn.rstrip('.txt')).rstrip('.html') + '.txt'),'w').write('\n'.join(words))

if __name__=='__main__':
	docs_folder = 'test_documents/'
	out_folder = 'Catchphrases_extracted_by_D2V_BiGRU_CRF/'
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)
	models_folder = 'The_D2V_BiGRU_CRF_model'
	d2v_modelname = 'D2Vmodel_for_insc_33545'
	get_annotations(docs_folder, out_folder, models_folder, d2v_modelname)
