import data_utils
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn_size
import time
import sys

class RNN_model():
	def __init__(self, num_classes, seq_len=50, batch_size=32, rnn_size=128, learning_rate=0.0005,\
				 num_epoch=25, num_layer=2, num_char=500, num_gen=100, in_dropout=0.8,\
				 predict_every_x_epoch=10):
		self.num_classes = num_classes
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.rnn_size = rnn_size
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.num_layer = num_layer
		self.num_char = num_char
		self.num_gen = num_gen
		self.in_dropout = in_dropout
		self.predict_every_x_epoch = predict_every_x_epoch

	def create_vars(self):
		self.x = tf.placeholder('float', [None, self.seq_len, self.num_classes])
		self.y = tf.placeholder('float', [None, self.num_classes])
		self.drop_placeholder = tf.placeholder_with_default(1.0, shape=())

		with tf.variable_scope('rnn'):
			self.W = tf.get_variable('W', [self.rnn_size, self.num_classes])
			self.b = tf.get_variable('b', [self.num_classes])

	def create_model(self, x, dropout=1, predict=False):
		if predict==False:
			self.create_vars()
		lstm_cell = [rnn.LayerNormBasicLSTMCell(self.rnn_size, activation=tf.nn.relu,\
					reuse=tf.AUTO_REUSE, dropout_keep_prob=dropout) for _ in range(self.num_layer)]
		lstm_cell = rnn.MultiRNNCell(lstm_cell)
		outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
		
		#output shape is (train size, seq len, n classsses)
		#unstack turns it into list with length seq len, each with shape (train size, n classes)
		outputs = tf.unstack(outputs, self.seq_len, 1)
		return tf.matmul(outputs[-1], self.W) + self.b

	def train_model(self, train_x, train_y):
		self.training_size = train_x.shape[0]
		model_logits = self.create_model(train_x, self.in_dropout)
		pred = tf.nn.softmax(model_logits)

		#define the loss and optimizer
		#sigmoid returns tensors which consist of the loss of each data
		#reduce mean is just averaging all that tensor's value
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y, logits=model_logits))
		loss_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

		#calculate acc
		correct_pred = tf.equal(tf.argmax(train_y, 1), tf.argmax(pred, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		saver = tf.train.Saver()

		with tf.Session() as sess:
			#initialize all variable
			sess.run(tf.global_variables_initializer())

			for i in range(1, self.num_epoch+1):
				start_time = time.time()
				epoch_loss = 0
				start = 0
				epoch_acc = 0
				for _ in range(int(self.training_size/self.batch_size)):
					end = self.batch_size
					epoch_x = train_x[start:end]
					epoch_y = train_y[start:end]

					#train on sess
					# _, c = sess.run([loss_op, loss], feed_dict={x: epoch_x, y:epoch_y})
					_, acc, c = sess.run([loss_op, accuracy, loss], feed_dict={self.x: epoch_x, self.y:epoch_y, self.drop_placeholder:self.in_dropout})

					start += self.batch_size
					end += self.batch_size
					epoch_loss += c
					epoch_acc += acc

				end_time = time.time()
				elapsed = end_time - start_time
				epoch_acc = epoch_acc * 100 / int(self.training_size/self.batch_size)
				elapsed = format(elapsed, '.4f')
				elapsed = str(elapsed) + 's'
				print('Epoch: {} | Loss: {:.4f} | Acc: {:.4f} | Elapsed: {}'.format(i, epoch_loss, epoch_acc, elapsed))
				
				#GENERATE THE TEXT
				if i%self.predict_every_x_epoch == 0:
					pred_mat = data_utils.generate_sample(self.seq_len)
					text = ''
					yhat_op = self.generate_prediction(pred_mat)
					yhat = sess.run(yhat_op, feed_dict={self.x: pred_mat, self.drop_placeholder:self.in_dropout})
					
					for j in range(self.num_gen):
						text += data_utils.indices_to_char(yhat[j])
						if j == self.num_gen-1:
							print(text)
							text = ''


			# Save model
			print('Finished training. Saving model...')
			save_path = saver.save(sess, './ckpt/model.ckpt')
			print('Model saved in {}'.format(save_path))

	def generate_prediction(self, x_gen):
		yhat_list = []
		for i in range(self.num_gen):
			out_z = self.create_model(x_gen, predict=True)
			yhat = tf.nn.softmax(out_z)

			#create one hot encoding of from the argmax
			yhat_argmax = tf.one_hot(tf.argmax(yhat[0]), depth=self.num_classes)
			yhat_list.append(yhat_argmax)

			#concatenate new yhat
			conc = tf.concat((x_gen, [[yhat_argmax]]), axis=1)
			#remove the first input
			x_gen = tf.slice(conc, [0, 1, 0], [1, self.seq_len, -1])

		#returns a list of characters which are represented by a matrix
		return yhat_list