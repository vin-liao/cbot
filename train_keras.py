import numpy as np
from utils import Data
import keras.optimizers
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNLSTM, Dropout, BatchNormalization, Activation
import getopt
import sys
from sklearn.utils import shuffle

#prediction
def predict(use_empty_vector=False):
	if use_empty_vector:
		#returns a zero vectors
		eval_matrix = data.generate_sample(zero_vectors=True)
		yhat_matrix = np.zeros((1, num_classes))
		text = ''
		for i in range(num_gen):
			if i == 0:
				yhat_matrix[0, np.random.choice(eval_matrix.shape[2])] = 1	

			else:
				yhat_matrix_raw = model.predict(eval_matrix)
				yhat_matrix[0, np.argmax(yhat_matrix_raw)] = 1

				yhat_char = data.indices_to_char(yhat_matrix)
				text += yhat_char

				eval_matrix = np.delete(eval_matrix, 0, axis=1)
				eval_matrix = np.append(eval_matrix, [yhat_matrix], axis=1)

				yhat_matrix[0, np.argmax(yhat_matrix_raw)] = 0

		print(text)

def main():
	seq_len = 50
	num_epoch = 25
	batch_size = 32
	num_char = 100
	num_gen = 20

	try:
	    opts, args = getopt.getopt(sys.argv[1:], 's:c:e:b:g:', ['seq=', 'char=', 'epoch=', 'batch=', 'gen='])
	except getopt.GetoptError:
	    sys.exit(2)

	for opt, arg in opts:
		if opt in ('-s', '--seq'):
			seq_len = int(arg)
		elif opt in ('-c', '--char'):
			num_char = int(arg)
		elif opt in ('-e', '--epoch'):
			num_epoch = int(arg)
		elif opt in ('-b', '--batch'):
			batch_size = int(arg)
		elif opt in ('-g', '--gen'):
			num_gen = int(arg)
		else:
			sys.exit(2)

	data = Data.Data(seq_len, hm_char=num_char)
	x, y = data.generate_data()
	num_classes = x.shape[2]
	x, y = shuffle(x, y, random_state=42)
	generate_callback = LambdaCallback(on_epoch_end=lambda batch, logs: predict(use_empty_vector=True))

	model = Sequential()
	model.add(CuDNNLSTM(200, input_shape=(seq_len, num_classes), return_sequences=True))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.5))

	model.add(CuDNNLSTM(200))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(.5))

	model.add(Dense(num_classes, activation='softmax'))

	adam = keras.optimizers.Adam(lr=0.001, clipvalue=1000)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())

	model.fit(x, y, epochs=num_epoch, batch_size=batch_size, verbose=1, callbacks=[generate_callback])

	print('Saving model...')
	model.save_weights('../char_rnn_model_weights.h5')
	print('Finished saving!')

if __name__ == '__main__':
	main()