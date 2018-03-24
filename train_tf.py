import data_utils
import model
import sys
import getopt

seq_len = 50
num_epoch = 25
batch_size = 32
num_char = 500

try:
    opts, args = getopt.getopt(sys.argv[1:], 's:c:e:b:', ['seq=', 'char=', 'epoch=', 'batch='])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
	if opt in ('-s', '--seq'):
		seq_len = arg
	elif opt in ('-c', '--char'):
		num_char = arg
	elif opt in ('-e', '--epoch'):
		num_epoch = arg
	elif opt in ('-b', '--batch'):
		batch_size = arg
	else:
		sys.exit(2)

x, y = data_utils.generate_data(seq_len, num_char)
num_classes = x.shape[2]
model = model.RNN_model(num_classes=num_classes,\
						num_epoch=num_epoch,\
						batch_size=batch_size,\
						num_char=num_char)
model.train_model(x, y)