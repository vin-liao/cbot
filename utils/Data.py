import numpy as np
import re

class Data():
	def __init__(self, seq_len, hm_char=10000, skip_char=1):
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
		self.sentences = []
		self.next_char = []
		self.text = ''
		self.seq_len = seq_len
		self.hm_char = hm_char
		self.skip_char = skip_char

	def generate_data(self, save_sentence=0):
		with open('./data/linux.txt', 'r') as f:
			for line in f:
				self.text += line.lower()
			f.close

		#hm_char -1 uses all data
		if self.hm_char == -1:
			pass
		else:
			self.text = self.text[:self.hm_char]

		self.chars = sorted(list(set(self.text)))
		self.char_len = len(self.chars)
		self.train_size = self.char_len-self.seq_len

		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

		#create sentences and their next character
		for i in range(0, len(self.text)-self.seq_len, self.skip_char):
			self.sentences.append(self.text[i:i+self.seq_len])
			self.next_char.append(self.text[i+self.seq_len])

		#map each character to one hot
		self.x = np.zeros((len(self.sentences), self.seq_len, len(self.chars)), dtype=np.float32)
		self.y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.float32)
		for i, sentence in enumerate(self.sentences):
			for j, char in enumerate(sentence):
				self.x[i, j, self.char_indices[char]] = 1
			self.y[i, self.char_indices[next_char[i]]] = 1

		if save_sentence != 0:
			write_sentence = np.random.choice(sentence, save_sentence, replace=False)
			with open('sentences.txt', 'w') as r:
				for one_sentence in write_sentence:
					r.write(write_sentence+'\n')

		return self.x, self.y

	def generate_sample(self, zero_vectors=False):
		if zero_vectors:
			if len(self.sentences) == 0:
				generate_data(self.seq_len, self.hm_char)
			return np.zeros((1, self.seq_len, self.char_len), dtype=np.float32)

		else:
			if len(self.sentences) == 0:
				generate_data(self.seq_len, self.hm_char)

			sample_matrix = np.zeros((1, self.seq_len, self.char_len), dtype=np.float32)
			seed = np.random.randint(len(self.sentences))	
			one_sentence = self.sentences[seed]

			for i, char in enumerate(one_sentence):
				sample_matrix[0, i, self.char_indices[char]] = 1

			return sample_matrix

	def char_to_indices(self, char):
		mat = np.zeros(self.char_len, dtype=np.float32)
		mat[self.char_indices[char]] = 1
		return mat

	def indices_to_char(self, mat):
		idx = np.where(mat[0]==1)
		return self.indices_char[idx[0][0]]