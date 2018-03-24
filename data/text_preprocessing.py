import re

with open('./linux_raw.txt', 'r+') as f:
	text = f.read()

	#remove comment
	text = re.sub('/\*.*?\*/', '', text, flags=re.DOTALL)
	text = re.sub('//.*?\n', '', text)

	f.seek(0)
	f.write(text)
	f.truncate()