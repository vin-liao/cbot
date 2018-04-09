import numpy as np
import re
import pickle

class Data():
	def __init__(self, seq_len, hm_char=10000, skip_char=1):
		'''
		seq_len: Sequence length.
		
		hm_char: How many char is taken from the text. the default is 10k, so only
				 the first 10k characters is used

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

		with open('./data/linux.txt', 'r') as f:
			if hm_char == -1:
				#use all data, this is for generator
				self.text = f.read()
			else:
				self.text = f.read(self.hm_char)
			f.close()

		self.chars = sorted(list(set(self.text)))
		self.char_len = len(self.chars)

		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
		self.num_classes = len(self.char_indices)

		#create sentences and their next character
		for i in range(0, len(self.text)-self.seq_len, self.skip_char):
			self.sentences.append(self.text[i:i+self.seq_len])
			self.next_char.append(self.text[i+self.seq_len])

		self.training_size = len(self.sentences)

	def generate_data(self, save_sentence=0):
		x, y = self.sentences_to_matrix(len(self.sentences), self.sentences, self.next_char)

		if save_sentence > 0:
			self.save_things(save_sentence)

		return x, y

	def generate_sample(self, zero_vectors=False):
		if zero_vectors:
			return np.zeros((1, self.seq_len, self.char_len), dtype=np.float32)

		else:
			if len(self.sentences) == 0:
				generate_data(self.seq_len, self.hm_char)

			sample_matrix = np.zeros((1, self.seq_len, self.char_len), dtype=np.float32)
			seed = np.random.randint(len(self.sentences))	
			one_sentence = self.sentences[seed]
			print('\nStarting char:')
			print(one_sentence)

			for i, char in enumerate(one_sentence):
				sample_matrix[0, i, self.char_indices[char]] = 1

			return sample_matrix

	def predict(self, model, num_gen=100, use_empty_vector=False):
		if use_empty_vector:
			eval_matrix = self.generate_sample(zero_vectors=True)
		else:
			eval_matrix = self.generate_sample()

		yhat_matrix = np.zeros(self.num_classes)
		text = ''

		for i in range(num_gen):
			if use_empty_vector and i == 0:
				yhat_matrix[np.random.choice(eval_matrix.shape[2])] = 1	

			else:
				yhat_matrix_raw = model.predict(eval_matrix)
				yhat_matrix[np.argmax(yhat_matrix_raw)] = 1

				yhat_char = self.indices_to_char(yhat_matrix)
				text += yhat_char

				eval_matrix = np.delete(eval_matrix, 0, axis=1)
				eval_matrix = np.append(eval_matrix, [[yhat_matrix]], axis=1)

				yhat_matrix[np.argmax(yhat_matrix_raw)] = 0
		
		print('Prediction:')	
		print(text)

	def get_generator(self, batch_size=32):
		while True:
			sentence_list = []
			target_list = []
			for i in range(batch_size):
				seed = np.random.choice(len(self.sentences))
				sentence_list.append(self.sentences[seed])
				target_list.append(self.next_char[seed])

			yield self.sentences_to_matrix(batch_size, sentence_list, target_list)

	def sentences_to_matrix(self, training_size, sentences, next_char):
		x = np.zeros((training_size, self.seq_len, len(self.chars)), dtype=np.float32)
		y = np.zeros((training_size, len(self.chars)), dtype=np.float32)
		for i, sentence in enumerate(sentences):
			for j, char in enumerate(sentence):
				x[i, j, self.char_indices[char]] = 1
			y[i, self.char_indices[next_char[i]]] = 1

		return x, y

	def char_to_indices(self, char):
		mat = np.zeros(self.char_len, dtype=np.float32)
		mat[self.char_indices[char]] = 1
		return mat

	def indices_to_char(self, mat):
		'''
		idx = tuples, i.e. (array([NUMBER]),)
		idx[0] = a list with single number, i.e. [NUMBER]
		idx[0][0] = a number
		'''
		idx = np.where(mat==1)
		return self.indices_char[idx[0][0]]

	def save_things(self, save_sentence):
		with open('indices_char.pickle', 'wb') as f1:
			pickle.dump(self.indices_char, f1)

		with open('char_indices.pickle', 'wb') as f2:
			pickle.dump(self.char_indices, f2)
		
		#sentences
		write_sentence = np.random.choice(self.sentences, save_sentence, replace=False)
		with open('sentences.txt', 'w') as r:
			for one_sentence in write_sentence:
				one_sentence = re.sub(r'\n', '`n`', one_sentence)
				one_sentence = re.sub(r'\t', '`t`', one_sentence)
				r.write(one_sentence)
			r.close()

	def get_num_classes(self):
		return self.num_classes

	def get_training_size(self):
		return self.training_size