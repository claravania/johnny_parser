import os
import sys
import pdb

from collections import defaultdict


log_file = sys.argv[1]
word_vocab = sys.argv[2]

vocab_dir = 'vocab'

dependencies = ['nsubj', 'obj', 'iobj', 'nmod', 'amod']
pos_cat = ['NOUN', 'VERB', 'ADJ']


morph_dist = {}
num_preds = {}
num_errors = {}

for d in dependencies:
	morph_dist[d] = defaultdict(int)
	num_preds[d] = 0
	num_errors[d] = 0

for pos in pos_cat:
	morph_dist[pos] = defaultdict(int)
	num_preds[pos] = 0
	num_errors[pos] = 0	

pos_vocab = os.path.join(vocab_dir, 'pos_vocab.txt')
label_vocab = os.path.join(vocab_dir, 'label_vocab.txt')


def load_vocab(infile):
	id2item = defaultdict(str)
	with open(infile) as f:
		for line in f:
			line = line.strip()
			iid, item = line.split(' ', 1)
			id2item[int(iid)] = item
	return id2item


def preprocess(seq, dec=False):
	seq = seq.strip().replace(' ', '')
	seq = seq[1: len(seq) - 1].split(',')
	seq = list(filter(None, seq))
	if dec:
		seq = [int(x) - 1 for x in seq]
	else:
		seq = [int(x) for x in seq]
	return seq

w_vocab = load_vocab(word_vocab)
p_vocab = load_vocab(pos_vocab)
l_vocab = load_vocab(label_vocab)
pos_tags = list(p_vocab.values())



# extract morph vocab
morph_vocab = set()
for val in w_vocab.values():
	if '=' in val:
		morph_vocab.add(val)

lines = open(log_file).readlines()
batch_size = 256
feats = set()
feats.add('Lemma')

i = 0
read_batch = ''
while i < len(lines):
	line = lines[i]
	line = line.strip()

	if line == '':
		i += 1
		continue

	if line.startswith('test'):
		break

	if line.startswith('Batch'):
		read_batch = line.strip()
		print(read_batch)
		i += 1
		line = lines[i].strip()
		line = line[1:len(line) - 1]
		
		sent_ids = [int(x) for x in line.split(', ')]
		
		if batch_size != len(sent_ids):
			batch_size = len(sent_ids)

		# read attention vectors
		i += 1
		attn_vectors = lines[i:i+batch_size]
		sorted_attn_vectors = [x.strip() for _, x in sorted(zip(sent_ids, attn_vectors))]
		sorted_attn_vectors = [x[1:len(x) - 1] for x in sorted_attn_vectors]
		sorted_attn_vectors = [[int(x) for x in vec.split(' ')] for vec in sorted_attn_vectors]

		i += batch_size

		# process each batch of sentence
		sents = lines[i:i+batch_size]
		i += batch_size

		for sent_idx, sent in enumerate(sents):
			sent = sent.strip()
			morph_ids, pos_ids, arc_pred, lbl_pred, arc_gold, lbl_gold = sent.split(' ||| ')

			sent_morphs = []
			morph_ids = morph_ids.strip()
			morph_ids = morph_ids[2: len(morph_ids) - 2]

			for w in morph_ids.split('), ('):
				w = w.replace(')', '')
				w = w.replace('(', '')
				
				m_ids = [x.strip() for x in w.split(',')]
				m_ids = list(filter(None, m_ids))
				
				m_ids = [int(x) for x in m_ids]
				m_ids = [3] + m_ids + [4]
				sent_morphs.append(m_ids)

			pos_ids = preprocess(pos_ids)
			arc_pred = preprocess(arc_pred, dec=True)
			lbl_pred = preprocess(lbl_pred)
			arc_gold = preprocess(arc_gold, dec=True)
			lbl_gold = preprocess(lbl_gold)

			for p in range(len(arc_pred)):

				ap = arc_pred[p]  # position of the predicted arc
				lp = lbl_pred[p]  # index of the predicted label
				ag = arc_gold[p]  # position of the gold arc
				lg = lbl_gold[p]  # index of the gold label

				# head is ROOT
				if ap == -1:
					continue

				# if prediction is true
				if ap == ag and lp ==lg:
					# part of speech of the head and dependent
					pos_head = pos_ids[ap]
					pos_dep = pos_ids[p]

					dep_lbl = l_vocab[lp]

					# only check for dependencies that we care about
					if dep_lbl not in dependencies:
						continue

					num_preds[dep_lbl] += 1
					attn_idx = sorted_attn_vectors[sent_idx][p]

					if attn_idx < len(sent_morphs[ap]):
						highest_feat = sent_morphs[ap][attn_idx]
						mfeat = w_vocab[highest_feat]
						if mfeat not in morph_vocab:
							morph_dist[dep_lbl]['Lemma'] += 1
						else:	
							label, val = mfeat.split('=')
							morph_dist[dep_lbl][label] += 1
							feats.add(label)
					else:
						print(p)
						num_errors[dep_lbl] += 1
						print(sent_idx)
						print(sorted_attn_vectors[sent_idx])
						print(arc_pred)
						print(sent_morphs)
						print(attn_idx)
						print(len(sent_morphs[ap]))
						# pdb.set_trace()

feats = sorted(list(feats))
for d in dependencies:
	print('Label:', d)
	for feat in feats:
		v = morph_dist[d][feat]
		acc = 0
		if num_preds[d] > 0:
			acc = round(v * 100.0 / num_preds[d], 2)
		print(feat, acc)
	print(num_preds[d], num_errors[d])
	print('=' * 30)
	print()
			

	