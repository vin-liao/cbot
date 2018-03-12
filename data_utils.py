import numpy as np

def get_data(seq_len):
	with open('./data/c_syntax.txt', 'r') as f:
		text = f.read().lower()
		f.close

	text = text[:10009]
	print(text)
	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	#create sentences and their next character
	sentences = []
	next_char = []
	for i in range(0, len(text)-seq_len):
		sentences.append(text[i:i+seq_len])
		next_char.append(text[i+seq_len])

	#map each character to one hot
	x = np.zeros((len(sentences), seq_len, len(chars)))
	y = np.zeros((len(sentences), len(chars)))
	for i, sentence in enumerate(sentences):
		for j, char in enumerate(sentence):
			x[i, j, char_indices[char]] = 1
		y[i, char_indices[next_char[i]]] = 1

	x = x.astype(np.float32)
	y = y.astype(np.float32)

	return x, y
get_data(50)