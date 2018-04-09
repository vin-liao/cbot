import numpy as np
from utils import Data
import keras.optimizers
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNLSTM, Dropout, BatchNormalization, Activation, LSTM
import getopt
import sys
from sklearn.utils import shuffle

seq_len = 50
num_epoch = 25
batch_size = 32
num_char = 1000
num_gen = 20
use_generator = True

try:
    opts, args = getopt.getopt(sys.argv[1:], 's:c:e:b:g:u:', \
    	['seq=', 'char=', 'epoch=', 'batch=', 'gen=', 'use_g=='])
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
	elif opt in ('-u', '--use_g'):
		if arg == 1:
			use_generator = True
	else:
		sys.exit(2)

data = Data.Data(seq_len, hm_char=num_char)
num_classes = data.get_num_classes()
training_size = data.get_training_size()

if use_generator == False:
	x, y = data.generate_data()
	x, y = shuffle(x, y, random_state=42)

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
generate_callback = LambdaCallback(on_epoch_end=lambda batch, logs: \
	data.predict(model=model, num_gen=num_gen, use_empty_vector=False))

if use_generator:
	model.fit_generator(data.get_generator(), epochs=num_epoch, \
		steps_per_epoch=training_size/batch_size, callbacks=[generate_callback], verbose=1)
else:
	model.fit(x, y, epochs=num_epoch, batch_size=batch_size, verbose=1, callbacks=[generate_callback])

print('Saving model...')
model.save_weights('../char_rnn_model_weights.h5')
print('Finished saving!')