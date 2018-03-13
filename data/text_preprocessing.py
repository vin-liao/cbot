import re

with open('./linux.txt', 'r+') as f:
	text = f.read()

	#remove comment
	text = re.sub('/\*.*?\*/', '', text, flags=re.DOTALL)
	text = re.sub('//.*?\n', '', text)

	'''
	replacing whitespace, so that the NN can pick this up,
	later on the NN can generate it's own whitespace, so
	humans can read their code.

	Later, after the generated text is done, these newlines
	that the NN generated is reversed, so it's readable.

	i.e.
	`n` -> \n
	'''
	text = re.sub('\n', '`n`', text)
	text = re.sub('\r', '`r`', text)
	text = re.sub('\f', '`f`', text)
	text = re.sub('\t', '`t`', text)

	f.seek(0)
	f.write(text)
	f.truncate()