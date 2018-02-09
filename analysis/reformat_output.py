import os
import sys
import re


source_dir = sys.argv[1]
target_dir = sys.argv[2]

filenames = os.listdir(source_dir)

for fname in filenames:
	if fname.endswith('.txt'):
		with open(os.path.join(source_dir, fname)) as  f:
			fout = open(os.path.join(target_dir, fname), 'w')
			cont = False
			for line in f:
				line = line.strip()
				if line.startswith('[') or cont:
					re.sub('\s+', ' ', line)
					if line.endswith(']'):
						fout.write(line + '\n')
						cont = False
					else:
						fout.write(line + ' ')
						cont = True
				else:
					if line.endswith(']'):
						re.sub('\s+', ' ', line)
						cont = False
					fout.write(line + '\n')
			fout.close()
