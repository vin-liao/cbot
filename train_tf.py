import data_utils
import numpy as np
import tensorflow as tf
import model
import sys

try:
	hm_char = int(sys.argv[1]) #value from the terminal
except (IndexError, ValueError) as e:
	hm_char = 500
	print(e, 'Setting default char value to {}'.format(hm_char))
	pass

seq_len = 50

x, y = data_utils.generate_data(seq_len, hm_char)
num_classes = x.shape[2]
model = model.RNN_model(num_classes=num_classes)
model.train_model(x, y)