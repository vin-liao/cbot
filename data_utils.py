import numpy as np
import re

def get_data(seq_len):
	text = ''
	with open('./data/c_syntax.txt', 'r') as f:
		for line in f:
			text += line.lower()
		f.close

	text = re.sub('\r', '', text)
	text = re.sub('\n', '', text)
	text = re.sub('\t', '', text)

	text = text[:400000]

	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	#create sentences and their next character
	print(len(chars))

	sentences = []
	next_char = []
	for i in range(0, len(text)-seq_len):
		sentences.append(text[i:i+seq_len])
		next_char.append(text[i+seq_len])

	print('a')
	#map each character to one hot
	x = np.zeros((len(sentences), seq_len, len(chars)))
	y = np.zeros((len(sentences), len(chars)))
	for i, sentence in enumerate(sentences):
		for j, char in enumerate(sentence):
			x[i, j, char_indices[char]] = 1
		y[i, char_indices[next_char[i]]] = 1

	print('b')

	x = x.astype(np.float32)
	y = y.astype(np.float32)

	return x, y