import numpy as np
import re
import tensorflow as tf

sentences = []
next_char = []

def generate_data(seq_len, hm_char=10000):
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

	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	#create sentences and their next character
	for i in range(0, len(text)-seq_len, 3):
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

def generate_sample(seq_len, hm_char=None):
	if len(sentences) == 0:
		generate_data(seq_len, hm_char)

	sample_matrix = np.zeros((1, seq_len, char_len), dtype=np.float32)
	#THIS CODE CAN STILL BE BETTER BY THE WAY
	seed = np.random.randint(len(sentences))	
	one_sentence = sentences[seed]	

	# while one_sentence.startswith('`n`') == False:
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