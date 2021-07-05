import subprocess
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
import random

def my_tokenizer(text,gram,legal_stopwords, remove_stopwords,stem, remove_digits):
	ignore_list = ['\'']
	
	#converting stopwords to lowercase
	for i in range(len(legal_stopwords)):
			legal_stopwords[i] = legal_stopwords[i].lower().encode('utf-8','ignore')
	#Adding english stopwords from nltk corpus
	if remove_stopwords:
		legal_stopwords=legal_stopwords+stopwords.words('english')
	else:
		legal_stopwords = [u'']
	#Considering hyphenated words on the same line
	text = text.replace('-\n','')
	text = text.replace('.',' ')
	text = text.replace(',',' ')
	text = text.replace(';',' ')
	text = text.replace('?',' ')
	text = text.replace('-',' ')
	text = text.replace('(',' ')
	text = text.replace(')',' ')
	text = text.replace('\"',' ')
	# Converting text to lowercase
	text = text.lower()
	#Breaking the text into tokens or unigrams
	maxind=5
	ind=0
	while(ind<maxind):
		text=text.replace('  ',' ')
		ind+=1
	##Getting the list of proper nouns
	pnouns_lst=[u'']
	#pnouns_lst = get_list_of_proper_nouns(text)
	#removing stopwords
	
	lines=text.split('\n')
	fin_unigrams=[]
	for line in lines:
		unigrams=line.split(' ')
		for unigram in unigrams:
			if (unigram in legal_stopwords) or (unigram in pnouns_lst):
				continue
			if all((c.isalpha() or c.isdigit() or c in ignore_list) for c in unigram):	#Considering only alphabets and digits
				fin_unigrams.append(unigram)
	if '' in unigrams:
		unigrams=unigrams.remove('')
	#Stemming the unigrams with porter stemmer
	if stem:
		stemmer = PorterStemmer()
		fin_unigrams = [stemmer.stem(token) for token in fin_unigrams]
	# Now we have the unigrams in hand
	# We will now join certain unigrams to form n grams
	n_grams=[]
	ind=0
	maxind=len(fin_unigrams)
	while(ind<=maxind-gram):
		ctr=0
		n_gram=[]
		while(ctr<gram):
			n_gram.append(fin_unigrams[ind+ctr])
			ctr+=1
		n_grams.append(' '.join(n_gram))
		ind+=1
	## Now the variable n_grams will contain the list of n grams
	return n_grams


def chunkify(lst,n):
	return [lst[i::n] for i in xrange(n)]
	
def get_sents(text):
	
	
	
	sentence_re = r'''(?x)      # set flag to allow verbose regexps
	      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
	    | \w+(-\w+)*            # words with optional internal hyphens
	    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
	    | \.\.\.                # ellipsis
	    | [][.,;"'?():-_`]      # these are separate tokens
	'''
	"""
	sentence_re = r'''(?x)      # set flag to allow verbose regexps
	      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
	    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
	    | \.\.\.                # ellipsis
	    | [][.,;"'?():-_`]      # these are separate tokens
	'''
	"""
	tokenizer = RegexpTokenizer(sentence_re)
	
	
	sentences=[]
	exception_list=[' ','\n','-','\"','(',')']
	new_text=''
	dot_count = text.count('.')
	#reading the text and keeping those dots that correspond to a full stop only
	curr_pos=0
	chk_range=5
	for letter in text:
		if letter=='.':
			not_in_range=True
			st_point=curr_pos-chk_range
			if st_point<0:
				st_point=0
			chk_text=text[st_point:curr_pos+chk_range]
			if chk_text.count('.')==1:
				chk_text = chk_text.replace('.',' ')
				for lt in chk_text:
					if not(lt.isalpha() or lt.isdigit() or lt in exception_list):
						not_in_range = False
			else:
				not_in_range = False
			if not_in_range:
				new_text += '.'
		else:
			new_text += letter
		curr_pos+=1
	sentences = [i.strip() for i in new_text.split('.')]
	for i in range(dot_count):
		if '' in sentences:
			sentences.remove('')
	n_sents=[]
	for sent in sentences:
		n_sents.append(my_tokenizer(sent,1,[''], False, False, False))
		#n_sents.append(tokenizer.tokenize(sent))
	return n_sents

"""
Algo:
	1. input the files
	2. we divide the input files into test train set
	3. Now each of the test and train set will give rise to a corresponding file
	4. 

"""

def get_lines(catches, sent):	
	targets = ['o']*len(sent)
	lines=[]
	word_ind = 0
	max_word_ind = len(sent)
	while (word_ind < max_word_ind):
		
		for catch in catches:
			catch_len = len(catch.split(' '))
			if word_ind+catch_len <= max_word_ind:
				#print
				#print catch
				#print ' '.join( sent[word_ind:word_ind+catch_len] )
				if (' '.join( sent[word_ind : word_ind+catch_len] )) == catch:
					#print 'Hello'
					#targets[word_ind:word_ind+catch_len)] = ['B-legal']+['I-legal']*(catch_len-1)
					catch_ind=0
					while(catch_ind < catch_len):
						if(catch_ind==0):
							targets[word_ind+catch_ind]='B-legal'
						else:
							targets[word_ind+catch_ind]='I-legal'
						catch_ind+=1
		word_ind+=1
	word_ind=0
	while(word_ind<max_word_ind):
		pos_tags = [i[1] for i in nltk.pos_tag(sent)] # pos_tags of this sentence
		
		line = sent[word_ind]+' '+pos_tags[word_ind]+' '+targets[word_ind]
		lines += [line]
		word_ind+=1
	return lines

