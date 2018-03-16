import tensorflow as tf
import data_utils
import numpy as np
from tensorflow.contrib import rnn
from train_tf import rnn_size, n_classes, seq_len, hm_char, char_len

# check variable tensors inside checkpoint
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file("./ckpt/model.ckpt", tensor_name='', all_tensors=True, all_tensor_names=True)

'''
Generate text

randomly find whitespaces, which means it's from the start
take seq_len amount of characters, use it as input

e.g.

`n`#include <xxx.h>`n`#include <yyy.h>`n` ... up to seq_len chars

or

if(x==0){
	printf("hey");
}

the processed text becomes
`n`if(x==0){`n``t`printf("hey")};

and the generator will take `n``t`printf("hey") ... up to seq_len chars

ANOTHER OPTION:
pad left everything then put one char on the most right, use
that as input

'''

hm_gen_char = 50
x_gen = tf.placeholder('float', [1, seq_len, char_len])
y_gen = tf.placeholder('float', [None, char_len])

weights = {
	'out': tf.Variable(tf.random_normal([rnn_size, n_classes]), name='rnn_weights')
}

biases = {
	'out': tf.Variable(tf.random_normal([n_classes]), name='rnn_biases')
}

def generate_text(x_gen):
	x_gen = tf.unstack(x_gen, seq_len, 1)
	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x_gen, dtype=tf.float32)
	out_z = tf.matmul(outputs[-1], weights['out']) + biases['out']
	return tf.nn.softmax(out_z)
	
saver = tf.train.Saver()
text = ''
with tf.Session() as sess:
	yhat_op = generate_text(x_gen)
	saver.restore(sess, './ckpt/model.ckpt')
	sess.run(tf.global_variables_initializer())
	mat_eval = data_utils.generate_sample(seq_len, hm_char)
	# for i in range 50 #LOOP HERE
	for i in range(hm_gen_char):
		yhat = sess.run(yhat_op, feed_dict={x_gen: mat_eval})

		argmax_yhat = np.argmax(yhat[0])
		new_mat = np.zeros(n_classes, dtype=np.float32)
		new_mat[argmax_yhat] = 1
		mat_eval = np.delete(mat_eval, 0, axis=1)
		mat_eval = np.append(mat_eval, [[new_mat]], axis=1)
		
		text += data_utils.indices_to_char(new_mat)

	print(text)