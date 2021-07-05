import sys
import tensorflow as tf
import os
from wrap import Sequence
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


config = tf.ConfigProto()

def create_labelled_data(file):
	f=open(file)
	x_arr=[]
	y_arr=[]
	i=0
	x_arr.append([])
	y_arr.append([])
	for line in f:
		try:
			line= line.strip()
			if len(line)==0:
				x_arr.append([])
				y_arr.append([])
				i+=1
				continue
			line=line.split('\t')
			if len(line)==2:
				x_arr[i].append(line[0])
				y_arr[i].append(line[1])
		except Exception as e:
			print(e)
	return x_arr,y_arr

def sanity_check(a,b):
	if len(a)!=len(b):
		return False

	for i,j in zip(a,b):
		if len(i)!=len(j):
			return False
	return True


def train(T):
	T=int(T)
	x_train, y_train= create_labelled_data('anago_train_test/train_'+str(T+1)+'.txt')
	#x_train, y_train= create_labelled_data('baby_trainfile.txt')
	print(len(x_train),len(x_train[10]), len(y_train))
	if sanity_check(x_train,y_train)==False:
		print('Sanity check failed')
		exit()
	model= Sequence()
	model.fit(x_train,y_train, epochs=5, verbose=1, batch_size=20)
	model.save("model_weights_"+str(T),"model_params_"+str(T),"model_pre_proc_"+str(T))
	x_test, y_test= create_labelled_data('anago_train_test/test_'+str(T+1)+'.txt')

	print (len(x_test),len(x_test[0]),len(x_test[2]))
	f_score,prec,recall= model.score(x_test,y_test)
	probs = model.predict(x_test)
	print()
	print(T)
	print("Precision: {}\n Recall: {} \n F1-score: {}".format(f_score,prec,recall))
	print("The probs are:")
	print(probs, len(probs), len(probs[0]))
	#model = load("model_weights","model_params")

if __name__=='__main__':
	arguments = sys.argv[1:]
	assert len(arguments)==1
	i = arguments[0]
	train(i)
	# train(0)
