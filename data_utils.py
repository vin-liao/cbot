import numpy as np
import re
import tensorflow as tf

sentences = []
next_char = []

def generate_data(seq_len, hm_char=10000, skip_char=1):
	'''
	seq_len: Sequence length.
	
	hm_char: How many char is taken from the text. the default is 10k, so only
			 the first 10k characters is used

	return_generator: If True, returns a generator
					  If false, returns (train, target)

	skip_char: How many characters to skip in the training data. Defualt is 1, 
			   which means it doesn't skip.

			   e.g.
			   skip 1
			   The quick brown fox ju -> m
			   he quick brown fox jum -> p
			   e quick brown fox jump -> s

			   skip 3
			   The quick brown fox ju -> m
			    quick brown fox jumps -> 
			   ck brown fox jumps ove -> r

	'''

	#this usage of global variable is probably a bad programming practice
	global char_indices
	global indices_char
	global char_len

	text = ''
	with open('./data/linux.txt', 'r') as f:
		for line in f:
			text += line.lower()
		f.close

	#only use hm_char amount of character
	if hm_char == -1:
		pass
	else:
		text = text[:hm_char]

	chars = sorted(list(set(text)))
	char_len = len(chars)
	train_size = char_len-seq_len

	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	#create sentences and their next character
	for i in range(0, len(text)-seq_len, skip_char):
		sentences.append(text[i:i+seq_len])
		next_char.append(text[i+seq_len])

	#map each character to one hot
	x = np.zeros((len(sentences), seq_len, len(chars)), dtype=np.float32)
	y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
	for i, sentence in enumerate(sentences):
		for j, char in enumerate(sentence):
			x[i, j, char_indices[char]] = 1
		y[i, char_indices[next_char[i]]] = 1

	return x, y

def get_generator(seq_len, hm_char=10000, skip_char=1, sample_size=100):

	text = ''
	with open('./data/linux.txt', 'r') as f:
		for line in f:
			text += line.lower()
		f.close

	#only use hm_char amount of character
	if hm_char == -1:
		pass
	else:
		text = text[:hm_char]

	chars = sorted(list(set(text)))
	char_len = len(chars)
	train_size = char_len-seq_len

	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	batch_size = int(train_size/sample_size)

	#maybe problem in for loop
	#my understanding about generator is still wrong
	while True:
		#create generator
		for batch in range(batch_size):
			sentences.clear()
			next_char.clear()
			for i in range(batch*sample_size, (batch+1)*sample_size-seq_len, skip_char):
				sentences.append(text[i:i+seq_len])
				next_char.append(text[i+seq_len])

			#map each character to one hot
			x = np.zeros((len(sentences), seq_len, len(chars)), dtype=np.float32)
			y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
			for i, sentence in enumerate(sentences):
				for j, char in enumerate(sentence):
					x[i, j, char_indices[char]] = 1
				y[i, char_indices[next_char[i]]] = 1

			yield x, y

def generate_sample(seq_len, hm_char=None, zero_vectors=False):
	if zero_vectors:
		if len(sentences) == 0:
			generate_data(seq_len, hm_char)
			return np.zeros((1, seq_len, char_len), dtype=np.float32)

	else:
		if len(sentences) == 0:
			generate_data(seq_len, hm_char)

		sample_matrix = np.zeros((1, seq_len, char_len), dtype=np.float32)
		#THIS CODE CAN STILL BE BETTER BY THE WAY
		seed = np.random.randint(len(sentences))	
		one_sentence = sentences[seed]	

		# while one_sentence.endswith('\n') == False:
		# 	seed = np.random.randint(len(sentences))	
		# 	one_sentence = sentences[seed]

		for i, char in enumerate(one_sentence):
			sample_matrix[0, i, char_indices[char]] = 1

		return sample_matrix

def char_to_indices(char):
	mat = np.zeros(char_len, dtype=np.float32)
	mat[char_indices[char]] = 1
	return mat

def indices_to_char(mat):
	idx = np.where(mat[0]==1)
	return indices_char[idx[0][0]]