import os
import sys
import pdb

from collections import defaultdict


vocab_dir = 'vocab'
lang = ['tr', 'fi', 'de', 'cs', 'ru']
# lang = ['cs', 'de', 'ru']
dependencies = ['nsubj', 'obj', 'iobj', 'nmod', 'amod', 'obl']
pos_cat = ['NOUN', 'VERB', 'ADJ']

for l in lang:
	log_file = 'new_logs/log_' + l + '_lbl.txt'
	word_vocab = 'vocab/word_vocab_attn.' + l

	print('Language:', l)
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


	def load_word_vocab(infile):
		id2item = defaultdict(str)
		morph_vocab = set()
		with open(infile) as f:
			for line in f:
				line = line.strip()
				iid, item = line.split(' ', 1)
				id2item[int(iid)] = item
				if '=' in item:
					lab, val = item.split('=')
					if lab != '':
						morph_vocab.add(lab)
		morph_vocab.add('Lemma')
		return id2item, morph_vocab


	def preprocess(seq, dec=False):
		seq = seq.strip().replace(' ', '')
		seq = seq[1: len(seq) - 1].split(',')
		seq = list(filter(None, seq))
		if dec:
			seq = [int(x) - 1 for x in seq]
		else:
			seq = [int(x) for x in seq]
		return seq

	w_vocab, morph_labels = load_word_vocab(word_vocab)
	p_vocab = load_vocab(pos_vocab)
	l_vocab = load_vocab(label_vocab)
	pos_tags = list(p_vocab.values())

	lines = open(log_file).readlines()
	batch_size = 256

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
			i += 2
			line = lines[i].strip()
			line = line[1:len(line) - 1]
			
			sent_ids = [int(x) for x in line.split(', ')]
			
			if batch_size != len(sent_ids):
				batch_size = len(sent_ids)

			# read attention vectors
			i += 2
			attn_vectors = lines[i:i+batch_size]
			sorted_attn_vectors = [x.strip().replace('\s+', '\s') for _, x in sorted(zip(sent_ids, attn_vectors))]
			sorted_attn_vectors = [x[1:len(x) - 1] for x in sorted_attn_vectors]
			sorted_attn_vectors = [[int(x) for x in vec.split()] for vec in sorted_attn_vectors]

			i += batch_size
			i += 1

			u_lbl_preds = [line.strip() for line in lines[i:i+batch_size]]
			i += batch_size
			i += 1

			# process each batch of sentence
			sents = lines[i:i+batch_size]
			i += batch_size

			for idx, sent in enumerate(sents):
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

				u_arc_pred = u_lbl_preds[idx].replace('\s+', '\s')
				u_arc_pred = u_arc_pred[1:len(u_arc_pred)-1].split()
				u_arc_pred = [int(x)-1 for x in u_arc_pred]

				for p in range(len(arc_pred)):

					ap = arc_pred[p]  # position of the predicted head
					lp = lbl_pred[p]  # index of the predicted label
					ag = arc_gold[p]  # position of the gold head
					lg = lbl_gold[p]  # index of the gold label

					u_ap = u_arc_pred[p]

					# head is ROOT
					if ap == -1:
						continue

					# if prediction is true
					if ap == ag and lp == lg and u_ap == ap:
						# part of speech of the head and dependent
						pos_head = pos_ids[ap]
						pos_dep = pos_ids[p]

						# check if child is a NOUN
						if p_vocab[pos_ids[p]] != 'NOUN':
							continue

						dep_lbl = l_vocab[lp]

						# only check for dependencies that we care about
						if dep_lbl not in dependencies:
							continue

						# filter words without analysis (lemma only representation)
						# if len(sent_morphs[ap]) <= 1:
						# 	continue

						# check the case of the child
						case = True
						# for feat_id in sent_morphs[p]:
						# 	feat = w_vocab[feat_id]
						# 	if '=' in feat:
						# 		label, val = feat.split('=')
						# 		if label.lower() == 'case' and val.lower() == 'dat':
						# 			case = True

						if case:

							num_preds[dep_lbl] += 1

							attn_idx = sorted_attn_vectors[idx][p]

							if attn_idx < len(sent_morphs[ap]): 
								feat_idx = sent_morphs[ap][attn_idx]
								mfeat = w_vocab[feat_idx]
								if '=' not in mfeat:
									morph_dist[dep_lbl]['Lemma'] += 1
								else:	
									label, val = mfeat.split('=')
									morph_dist[dep_lbl][label] += 1
							else:
								num_errors[dep_lbl] += 1

	feats = sorted(list(morph_labels))

	print('NA', ' '.join(feats))
	for d in dependencies:
		if num_preds[d] == 0:
			continue
		stat = [d]
		for feat in feats:
			v = morph_dist[d][feat]
			acc = 0
			if num_preds[d] > 0:
				acc = round(v * 100.0 / num_preds[d], 2)
				stat.append(str(acc))
		print(' '.join(stat))
	
	print()

	