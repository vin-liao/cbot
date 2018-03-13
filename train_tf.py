import data_utils
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import sys

try:
	hm_char = sys.argv[1] #value from the terminal
except IndexError:
	hm_char = 1000000 #default value

seq_len = 50
train_x, train_y = data_utils.get_data(seq_len, hm_char)
char_len = train_x.shape[2]
training_size = train_x.shape[0]
batch_size = 32
rnn_size = 128
n_classes = char_len
learning_rate = 0.001
hm_epoch = 100

x = tf.placeholder('float', [None, seq_len, char_len])
y = tf.placeholder('float', [None, char_len])

def create_model(x):
	weights = {
		'out': tf.Variable(tf.random_normal([rnn_size, n_classes]))
	}

	biases = {
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	"""
	current input = (batch size, sequence length, num input)
	wanted rnn input = list of seq len with shape (batch size, num input)
	"""

	x = tf.unstack(x, seq_len, 1)
	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

model_logits = create_model(train_x)
pred = tf.nn.sigmoid(model_logits)

#define the loss and optimizer
#sigmoid returns tensors which consist of the loss of each data
#reduce mean is just averaging all that tensor's value
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_y, logits=model_logits))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#calculate acc
correct_pred = tf.equal(tf.argmax(train_y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
	#initialize all variable
	sess.run(tf.global_variables_initializer())

	for i in range(1, hm_epoch+1):
		start_time = time.time()
		epoch_loss = 0
		start = 0
		for _ in range(int(training_size/batch_size)):
			end = batch_size
			epoch_x = train_x[start:end]
			epoch_y = train_y[start:end]


			#train on sess
			_, c = sess.run([train_op, loss], feed_dict={x: epoch_x, y:epoch_y})

			start += batch_size
			end += batch_size
			epoch_loss += c

		end_time = time.time()
		elapsed = end_time - start_time
		elapsed = str(elapsed) + 's'
		print('Epoch: ', i, ' Loss: ', epoch_loss, 'Elapsed: ', elapsed)

	#TODO evaluate model here