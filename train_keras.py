import numpy as np
import data_utils
from keras.models import Sequential
import keras.optimizers
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNLSTM, Dropout, BatchNormalization, Activation, LSTM
from sklearn.model_selection import train_test_split
import getopt
import sys

seq_len = 50
num_epoch = 25
batch_size = 32
num_char = 500
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

#prediction
def predict():
	eval_matrix = data_utils.generate_sample(seq_len, num_char)
	yhat_matrix = np.zeros((1, eval_matrix.shape[2]))
	text = ''
	for _ in range(num_gen):
		yhat_matrix_raw = model.predict(eval_matrix)
		yhat_matrix[0, np.argmax(yhat_matrix_raw)] = 1

		yhat_char = data_utils.indices_to_char(yhat_matrix)
		text += yhat_char

		eval_matrix = np.delete(eval_matrix, 0, axis=1)
		eval_matrix = np.append(eval_matrix, [yhat_matrix], axis=1)

		yhat_matrix[0, np.argmax(yhat_matrix_raw)] = 0

	print(text)

x, y = data_utils.generate_data(seq_len, num_char)
num_classes = x.shape[2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
generate_callback = LambdaCallback(on_epoch_end=lambda batch, logs: predict())

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(seq_len, num_classes), return_sequences=True))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(.7))

model.add(CuDNNLSTM(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(num_classes, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.001, clipvalue=1000)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epoch, batch_size=batch_size, verbose=2, callbacks=[generate_callback])
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print('Saving model...')
model.save('../char_rnn_model.h5')
print('Finished saving!')