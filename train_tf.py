import data_utils
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import sys

try:
	hm_char = int(sys.argv[1]) #value from the terminal
except (IndexError, ValueError) as e:
	hm_char = 100
	print(e, 'Setting default char value to {}'.format(hm_char))
	pass

seq_len = 50
train_x, train_y = data_utils.generate_data(seq_len, hm_char)
n_classes = train_x.shape[2]
training_size = train_x.shape[0]
batch_size = 32
rnn_size = 128
learning_rate = 0.001
hm_epoch = 100
hm_gen = 200
predict_every_x_epoch = 10

x = tf.placeholder('float', [None, seq_len, n_classes])
y = tf.placeholder('float', [None, n_classes])

with tf.variable_scope('rnn'):
	W = tf.get_variable('W', [rnn_size, n_classes])
	b = tf.get_variable('b', [n_classes])

def create_model(x):
	"""
	current input = (batch size, sequence length, num input)
	wanted rnn input = list of seq len with shape (batch size, num input)
	"""

	#this graph is only needed to be created once. If I want to predict, I
	#don't want to re-create a new graph, that's why reuse=tf.AUTO_REUSE
	#if reuse=False, new graph is created everytime this function is called
	lstm_cell = [rnn.BasicLSTMCell(rnn_size, reuse=tf.AUTO_REUSE, activation=tf.nn.relu) for _ in range(2)]
	lstm_cell = rnn.MultiRNNCell(lstm_cell)
	outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
	
	#output shape is (train size, seq len, n classes)
	#unstack turns it into list with length seq len, each with shape (train size, n classes)
	outputs = tf.unstack(outputs, seq_len, 1)

	return tf.matmul(outputs[-1], W) + b

def generate_yhat(x_gen):
	yhat_list = []
	for i in range(hm_gen):
		out_z = create_model(x_gen)
		yhat = tf.nn.softmax(out_z)

		#create one hot encoding of from the argmax
		yhat_argmax = tf.one_hot(tf.argmax(yhat[0]), depth=n_classes)
		yhat_list.append(yhat_argmax)

		#concatenate new yhat
		conc = tf.concat((x_gen, [[yhat_argmax]]), axis=1)
		#remove the first input
		x_gen = tf.slice(conc, [0, 1, 0], [1, seq_len, -1])

	#returns a list of characters which are represented by a matrix
	return yhat_list
	
def train_rnn():
	model_logits = create_model(train_x)
	pred = tf.nn.softmax(model_logits)

	#define the loss and optimizer
	#sigmoid returns tensors which consist of the loss of each data
	#reduce mean is just averaging all that tensor's value
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y, logits=model_logits))
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	#calculate acc
	correct_pred = tf.equal(tf.argmax(train_y, 1), tf.argmax(pred, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	saver = tf.train.Saver()

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
			elapsed = format(elapsed, '.4f')
			elapsed = str(elapsed) + 's'
			print('Epoch: {} | Loss: {:.4f} | Elapsed: {}'.format(i, epoch_loss, elapsed))
			
			#GENERATE THE TEXT
			#do this every 5 epoch
			if i%predict_every_x_epoch == 0:
				pred_mat = data_utils.generate_sample(seq_len)
				text = ''
				yhat_op = generate_yhat(pred_mat)
				yhat = sess.run(yhat_op, feed_dict={x: pred_mat})
				
				for j in range(hm_gen):
					text += data_utils.indices_to_char(yhat[j])
					if j == hm_gen-1:
						print(text)
						text = ''


		# Save model
		print('Finished training. Saving model...')
		save_path = saver.save(sess, './ckpt/model.ckpt')
		print('Model saved in {}'.format(save_path))

if __name__ == '__main__':
	train_rnn()