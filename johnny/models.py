import six
import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer
from time import sleep
from chainer import Variable, cuda
from johnny.utils import bar, discrete_print
from johnny.extern import DependencyDecoder
from johnny.vocab import UDepVocab


import pdb


# TODO Check multiple roots issue
# TODO Check constraint algorithms + optimise
# TODO Maybe switch to using predicted arcs towards end of training
# TODO Think of ways of avoiding self prediction
# Clara: TODO think about different MLP representation for words (see Dozat, et al., 2017)
class GraphParser(chainer.Chain):

    MIN_PAD = -100.
    TREE_OPTS = ['none', 'chu', 'eisner']

    def __init__(self,
                 encoder,
                 num_labels=46,
                 num_aux_lbls=0,
                 mlp_arc_units=100,
                 mlp_lbl_units=100,
                 mlp_tag_units=100,
                 arc_dropout=0.0,
                 lbl_dropout=0.5,
                 tag_dropout=0.2,
                 treeify='chu',
                 visualise=False,
                 debug=False,
                 sub_attn=False,
                 add_feat=False,
                 apply_mtl=False,
                 alpha=1.0,
                 beta=0.0
                 ):

        super(GraphParser, self).__init__()
        self.num_labels = num_labels
        self.num_aux_lbls = num_aux_lbls
        self.mlp_arc_units = mlp_arc_units
        self.mlp_lbl_units = mlp_lbl_units
        self.mlp_tag_units = mlp_tag_units
        self.arc_dropout = arc_dropout
        self.lbl_dropout = lbl_dropout
        self.tag_dropout = tag_dropout
        self.treeify = treeify.lower()
        self.visualise = visualise
        self.debug = debug
        self.sleep_time = 0.
        self.sub_attn = sub_attn
        self.add_feat = add_feat

        # MTL parameters
        self.apply_mtl = apply_mtl
        self.alpha = alpha
        self.beta = beta

        assert(treeify in self.TREE_OPTS)
        self.unit_mult = 2 if encoder.use_bilstm else 1

        with self.init_scope():

            self.encoder = encoder
            self.vT = L.Linear(mlp_arc_units, 1)

            # head
            self.H_arc = L.Linear(self.unit_mult*self.encoder.num_units, mlp_arc_units)
            
            # dependent
            self.D_arc = L.Linear(self.unit_mult*self.encoder.num_units, mlp_arc_units)

            if self.sub_attn:
                # parameters for subword attention
                self.units_dim = self.encoder.embedder.word_encoder.num_units
                self.max_sub_len = self.encoder.embedder.word_encoder.max_sub_len

                # k = softmax([sub_feats_head] x V_attn x h_dep)
                self.V_attn = L.Bilinear(self.units_dim*self.max_sub_len, mlp_arc_units, self.max_sub_len)
                
                # g = sigmoid(Wglob x h_head + m_head)
                self.W_glob = L.Linear(mlp_arc_units, mlp_arc_units)
                self.W_loc = L.Linear(mlp_arc_units, mlp_arc_units)

                # parameters for computing the scores
                self.W_head = L.Linear(mlp_arc_units, mlp_arc_units)
                self.W_dependent = L.Linear(mlp_arc_units, mlp_arc_units)
                self.U_lbl_attn = L.Linear(mlp_arc_units, mlp_lbl_units)

            if self.beta > 0:
                self.l1_tag = L.Linear(self.unit_mult*self.encoder.num_units, self.mlp_tag_units)
                self.l2_tag = L.Linear(self.mlp_tag_units, self.mlp_tag_units)
                self.out_tag = L.Linear(self.mlp_tag_units, self.num_aux_lbls)

                
            self.V_lblT = L.Linear(mlp_lbl_units, self.num_labels)
            self.U_lbl = L.Linear(self.unit_mult*self.encoder.num_units, mlp_lbl_units)
            self.W_lbl = L.Linear(self.unit_mult*self.encoder.num_units, mlp_lbl_units)


    def _predict_heads(self, sent_states, mask, batch_stats, sorted_heads=None):
        """For each token in the sentence predict which token in the sentence
        is its head."""

        batch_size, max_sent_len, col_lengths = batch_stats

        calc_loss = sorted_heads is not None


        # In order to predict which head is most probable for a given word
        # For each token in the sentence we get a vector represention
        # for that token as a head, and another for that token as a dependent.
        # The idea here is that we only need to calculate these once
        # and then reuse them to get all combinations
        # ------------------------------------------------------
        # In g(a_j, a_i) we note that we can precompute the matrix multiplications
        # for each word, we consider all possible heads
        # we can pre-calculate U * a_j , for all a_j
        # head activations for each token lstm activation
        h_arc = self.H_arc(sent_states)

        # we transform results to be indexable by sentence index for upcoming for loop
        # h_arc is now max_sent x bs x mlp_arc_units
        h_arc = F.reshape(h_arc, (-1, batch_size, self.mlp_arc_units))

        # bs * max_sent x mlp_arc_units
        d_arc = self.D_arc(sent_states)
        # max_sent x bs x mlp_arc_units
        d_arc = F.reshape(d_arc, (-1, batch_size, self.mlp_arc_units))

        # the values to use to mask softmax for head prediction
        # e ^ -100 is ~ zero (can be changed from self.MIN_PAD)
        mask_vals = Variable(self.xp.full((batch_size, max_sent_len),
                                           self.MIN_PAD,
                                           dtype=self.xp.float32))

        sent_arcs = []
        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            num_active = col_lengths[i]
            # if we are calculating loss create truth variables
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_heads = sorted_heads[i-1]
            # ================== HEAD PREDICTION ======================
            # NOTE Because some sentences may be shorter - only num_active of
            # the batch have valid activations for this token. If in softmax
            # we didn't limit arcs to [:num_active] we would need to replace
            # embeddings that are out of sentence range with zeros - because
            # otherwise when broadcasting and summing we will modify valid
            # batch activations for earlier tokens of the sentence.
            # ====================== Code for padding ==========================
            # invalid_pad = ((0, int(batch_size - num_active)), (0, 0))
            # d_arc_pad = F.pad(d_arc[i][:num_active],
            #                   invalid_pad, 'constant', constant_values=0.)
            # ==================================================================
            a_u, a_w = F.broadcast(h_arc, d_arc[i])

            arc_logit = F.reshape(F.tanh(a_u + a_w), (-1, self.mlp_arc_units))

            if self.arc_dropout > 0.:
                arc_logit = F.dropout(arc_logit, ratio=self.arc_dropout)

            arc_logit = self.vT(arc_logit)
            arcs = F.swapaxes(F.reshape(arc_logit, (-1, batch_size)), 0, 1)
            arcs = F.where(mask, arcs, mask_vals)
            # Calculate losses
            if calc_loss:
                # we don't want to average out over seen words yet
                # NOTE: do not use ignore_label - in gpu mode gold_heads gets mutated
                # and furthermore we would need to have padded invalid state of
                # d_arc[i] with zeros before broadcasting. 
                # see NOTE above
                head_loss = F.sum(F.softmax_cross_entropy(arcs[:num_active], gold_heads[:num_active], reduce='no'))
                self.loss += head_loss
            sent_arcs.append(F.reshape(arcs, (batch_size, -1, 1)))
        arcs = F.concat(sent_arcs, axis=2)

        return arcs


    def _predict_labels(self, sent_states, pred_heads, gold_heads, batch_stats,
                        sorted_labels=None):
        """Predict the label for each of the arcs predicted in _predict_heads."""

        batch_size, max_sent_len, col_lengths = batch_stats

        calc_loss = sorted_labels is not None
        if calc_loss:
            labels = self.encoder.transpose_batch(sorted_labels)

        u_lbl = self.U_lbl(sent_states)
        u_lbl = F.reshape(u_lbl, (-1, batch_size, self.mlp_lbl_units))

        w_lbl = self.W_lbl(sent_states)
        w_lbl = F.reshape(w_lbl, (-1, batch_size, self.mlp_lbl_units))

        sent_lbls = []
        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            # num_active
            num_active = col_lengths[i]
            # if we are calculating loss create truth variables
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_labels = labels[i-1]
                true_heads = gold_heads[i-1]

            arc_pred = pred_heads[i-1]

            # ================== LABEL PREDICTION ======================
            # TODO: maybe we should use arc_pred sometimes in training??
            # NOTE: gold_heads values after num_active gets mutated here
            # make sure you don't use ignore_label in softmax - even if ok in forward
            # it will be wrong in backprop (we limit to :self.num_active)
            # gh_copy = self.xp.copy(gold_heads.data)
            head_indices = true_heads.data if chainer.config.train else arc_pred
            head_indices = head_indices[:num_active]

            l_heads = u_lbl[head_indices, self.xp.arange(len(head_indices)), :]
            l_w = w_lbl[i][:num_active]

            UWl = F.reshape(F.tanh(l_heads + l_w), (-1, self.mlp_lbl_units))

            if self.lbl_dropout > 0.:
                UWl = F.dropout(UWl, ratio=self.lbl_dropout)

            lbls = self.V_lblT(UWl)

            # Calculate losses
            if calc_loss:
                label_loss = F.sum(F.softmax_cross_entropy(lbls[:num_active], gold_labels[:num_active], reduce='no'))
                self.loss += label_loss

            reshaped_lbls = F.reshape(lbls, (num_active, -1, 1))
            reshaped_lbls = F.pad(reshaped_lbls,
                    ((0, batch_size - num_active),(0,0), (0,0)), 'constant')
            sent_lbls.append(reshaped_lbls)
        lbls = F.concat(sent_lbls, axis=2)
        return lbls


    def _predict_heads_attn(self, sent_states, subword_embeds, mask, sub_lengths, batch_stats, extract_attn=False, sorted_heads=None):
        """For each token in the sentence predict which token in the sentence
        is its head."""

        # sub_lengths is the mask for morph embeddings -- refer to the subword lengths
        # start and end word are masked

        batch_size, max_sent_len, col_lengths = batch_stats

        calc_loss = sorted_heads is not None

        # In order to predict which head is most probable for a given word
        # For each token in the sentence we get a vector represention
        # for that token as a head, and another for that token as a dependent.
        # The idea here is that we only need to calculate these once
        # and then reuse them to get all combinations
        # ------------------------------------------------------
        # In g(a_j, a_i) we note that we can precompute the matrix multiplications
        # for each word, we consider all possible heads
        # we can pre-calculate U * a_j , for all a_j
        # head activations for each token lstm activation
        # bs * max_sent x mlp_arc_units
        h_arc = self.H_arc(sent_states)

        # we transform results to be indexable by sentence index for upcoming for loop
        # h_arc is now max_sent x bs x mlp_arc_units
        h_arc = F.reshape(h_arc, (-1, batch_size, self.mlp_arc_units))

        # bs * max_sent x mlp_arc_units
        d_arc = self.D_arc(sent_states)

        # max_sent x bs x mlp_arc_units
        d_arc = F.reshape(d_arc, (-1, batch_size, self.mlp_arc_units))

        # the values to use to mask softmax for head prediction
        # e ^ -100 is ~ zero (can be changed from self.MIN_PAD)
        mask_vals = Variable(self.xp.full((batch_size, max_sent_len),
                                           self.MIN_PAD,
                                           dtype=self.xp.float32))


        # reshape sub_lengths for attention computation
        # max_sent_len x bs x self.unit_mult*self.encoder.num_units
        sub_lengths = F.swapaxes(sub_lengths, axis1=0, axis2=1)

        # subword_embeds shape is bs x max_sen x max_sub_len x units_dim
        # reshape subword embeds to compute attention
        # max_sent x bs x units_dim*max_sub_len
        subword_embeds = F.reshape(subword_embeds, (-1, batch_size, self.units_dim*self.max_sub_len))  

        sent_arcs = []
        sent_attn_vectors = []
        sent_h_heads = []

        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):

            num_active = col_lengths[i]
            # if we are calculating loss create truth variables
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_heads = sorted_heads[i-1]
            # ================== HEAD PREDICTION ======================
            # NOTE Because some sentences may be shorter - only num_active of
            # the batch have valid activations for this token. If in softmax
            # we didn't limit arcs to [:num_active] we would need to replace
            # embeddings that are out of sentence range with zeros - because
            # otherwise when broadcasting and summing we will modify valid
            # batch activations for earlier tokens of the sentence.
            # ====================== Code for padding ==========================
            # invalid_pad = ((0, int(batch_size - num_active)), (0, 0))
            # d_arc_pad = F.pad(d_arc[i][:num_active],
            #                   invalid_pad, 'constant', constant_values=0.)
            # ==================================================================

            # h_i is the current word and h_j is the candidate head
            h_i = d_arc[i]

            # we compute the attention for every possible head
            h_heads = []
            attn_vectors = []

            # now, we start from 0 because we consider ROOT as head
            for j in range(max_sent_len):

                # f_j is the morph features of h_j
                f_j = subword_embeds[j]
                h_j = h_arc[j]

                # compute the attention vector
                k = self.V_attn(f_j, h_i)

                # or.. (another option)
                # k = self.V_attn(f_j, F.concat((h_i, h_j), axis=1))
                
                # attention mask, shape is the same as k
                attn_mask = Variable(self.xp.full(k.shape,
                                           self.MIN_PAD,
                                           dtype=self.xp.float32))

                # NOTE that we also put mask to the start and end symbol of the subword unit sequence
                cond = sub_lengths[j]
                k = F.where(cond, k, attn_mask)
                k = F.softmax(k, axis=1)

                attn_vectors.append(k)
                
                # reshape k and f_j to compute m_j
                k = F.reshape(k, (-1, 1))
                f_j = F.reshape(f_j, (-1, self.units_dim))

                # compute m_j
                m_j = f_j * k.data
                m_j = F.reshape(m_j, (batch_size, self.max_sub_len, self.units_dim))
                m_j = F.sum(m_j, axis=1)

                
                # compute gating function
                g = F.sigmoid(self.W_glob(h_j) + self.W_loc(h_i))
                z_j = g * h_j + (1 - g) * m_j

                h_heads.append(z_j)

            # sent_attn_vectors store the attention vectors for each dependent word in the batch
            # max_sent_len - 1 (because we don't consider root) x max_sent_len x bs x num of morph features
            sent_attn_vectors.append(attn_vectors)
            
            h_i = self.W_dependent(h_i)

            h_heads = self.W_head(F.reshape(F.stack(h_heads, axis=0), (-1, self.mlp_arc_units)))
            h_heads = F.reshape(h_heads, (-1, batch_size, self.mlp_arc_units))
            sent_h_heads.append(h_heads)

            a_u, a_w = F.broadcast(h_heads, h_i)

            arc_logit = F.reshape(F.tanh(a_u + a_w), (-1, self.mlp_arc_units))

            if self.arc_dropout > 0.:
                arc_logit = F.dropout(arc_logit, ratio=self.arc_dropout)

            arc_logit = self.vT(arc_logit)
            
            arcs = F.swapaxes(F.reshape(arc_logit, (-1, batch_size)), 0, 1)            
            arcs = F.where(mask, arcs, mask_vals)

            # Calculate losses
            if calc_loss:
                # we don't want to average out over seen words yet
                # NOTE: do not use ignore_label - in gpu mode gold_heads gets mutated
                # and furthermore we would need to have padded invalid state of
                # d_arc[i] with zeros before broadcasting. 
                # see NOTE above
                head_loss = F.sum(F.softmax_cross_entropy(arcs[:num_active], gold_heads[:num_active], reduce='no'))
                self.loss += head_loss

            sent_arcs.append(F.reshape(arcs, (batch_size, -1, 1)))

        arcs = F.concat(sent_arcs, axis=2)

        # sent_h_heads store the morphological representations
        # for each possible heads, for each dependent word
        # max_sent_len - 1 (num of dep) x max_sent_len (num of possible heads) x bs x mlp_arc_units
        # in other words, each word has different views/representation of its possible heads 
        sent_h_heads = F.stack(sent_h_heads, axis=0)

        return arcs, sent_h_heads, sent_attn_vectors


    def _predict_labels_attn(self, sent_states, pred_heads, pred_reps, sent_attn_vectors, gold_heads, batch_stats,
                        extract_attn=False, sorted_labels=None):
        """Predict the label for each of the arcs predicted in _predict_heads."""

        batch_size, max_sent_len, col_lengths = batch_stats

        calc_loss = sorted_labels is not None
        if calc_loss:
            labels = self.encoder.transpose_batch(sorted_labels)

        u_lbl = self.U_lbl(sent_states)
        u_lbl = F.reshape(u_lbl, (-1, batch_size, self.mlp_lbl_units))

        w_lbl = self.W_lbl(sent_states)
        w_lbl = F.reshape(w_lbl, (-1, batch_size, self.mlp_lbl_units))

        sent_lbls = []
        morph_heads = []

        # we start from 1 because we don't consider root
        for i in range(1, max_sent_len):
            # num_active
            num_active = col_lengths[i]
            # if we are calculating loss create truth variables
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_labels = labels[i-1]
                true_heads = gold_heads[i-1]

            arc_pred = pred_heads[i-1]
            head_reps = pred_reps[i-1]

            if extract_attn:
                max_feats = []
                # true_bs is the length of each column
                true_bs = len(arc_pred)
                # get attention vector
                for bs in range(batch_size):
                    if bs < true_bs:
                        head_idx = cuda.to_cpu(arc_pred[bs])
                        attn_matrix = sent_attn_vectors[i-1][head_idx]
                        max_feat = F.argmax(attn_matrix[bs], 0).data
                        max_feats.append(max_feat)
                    else:
                        max_feats.append(self.xp.array(0, dtype=self.xp.int32))
                morph_heads.append(F.stack(max_feats, 0))

            # ================== LABEL PREDICTION ======================
            # TODO: maybe we should use arc_pred sometimes in training??
            # NOTE: gold_heads values after num_active gets mutated here
            # make sure you don't use ignore_label in softmax - even if ok in forward
            # it will be wrong in backprop (we limit to :self.num_active)
            # gh_copy = self.xp.copy(gold_heads.data)
            head_indices = true_heads.data if chainer.config.train else arc_pred
            head_indices = head_indices[:num_active]

            l_heads = head_reps[head_indices, self.xp.arange(len(head_indices)), :]
            l_heads = self.U_lbl_attn(l_heads)
            l_w = w_lbl[i][:num_active]

            UWl = F.reshape(F.tanh(l_heads + l_w), (-1, self.mlp_lbl_units))

            if self.lbl_dropout > 0.:
                UWl = F.dropout(UWl, ratio=self.lbl_dropout)

            lbls = self.V_lblT(UWl)

            # Calculate losses
            if calc_loss:
                label_loss = F.sum(F.softmax_cross_entropy(lbls[:num_active], gold_labels[:num_active], reduce='no'))
                self.loss += label_loss

            reshaped_lbls = F.reshape(lbls, (num_active, -1, 1))
            reshaped_lbls = F.pad(reshaped_lbls,
                    ((0, batch_size - num_active),(0,0), (0,0)), 'constant')
            sent_lbls.append(reshaped_lbls)

        lbls = F.concat(sent_lbls, axis=2)

        if extract_attn:
            print('Attention vectors:')
            morph_heads = F.stack(morph_heads, axis=0)
            morph_heads = F.transpose(morph_heads)
            for dat in morph_heads.data:
                print(dat)

        return lbls


    def _predict_tags(self, sent_states, mask, batch_stats, sorted_tags=None):
        """For each token in the sentence predict its tag."""

        sorted_heads = sorted_tags
        batch_size, max_sent_len, col_lengths = batch_stats
        calc_loss = sorted_heads is not None

        h_tag = F.relu(self.l1_tag(sent_states))
        h_tag = F.dropout(F.relu(self.l2_tag(h_tag)), ratio=self.tag_dropout)

        # h_tag = self.l1_tag(sent_states)
        # h_tag = F.dropout(self.l2_tag(h_tag), ratio=self.tag_dropout)
        
        tag_logits = self.out_tag(h_tag)
        tag_logits = F.reshape(tag_logits, (-1, batch_size, self.num_aux_lbls))

        total_loss = 0
        total_acc = 0
        sent_tags = []
        for i in range(1, max_sent_len):
            if calc_loss:
                # i-1 because sentence has root appended to beginning
                gold_tags = sorted_heads[i-1]
                pred_tags = tag_logits[i]
                num_active = col_lengths[i]
                tag_loss = F.sum(F.softmax_cross_entropy(pred_tags[:num_active], gold_tags[:num_active], reduce='no'))
                tag_acc = F.sum(F.accuracy(pred_tags[:num_active], gold_tags[:num_active]))
                total_loss += tag_loss
                total_acc += tag_acc

            pred = F.argmax(F.softmax(pred_tags[:num_active]), axis=1)
            sent_tags.append(pred)

        total_acc = total_acc / (max_sent_len - 1)

        return total_loss, total_acc, sent_tags


    def __call__(self, flags, *inputs, **kwargs):
        """ Expects a batch of sentences 
        so a list of K sentences where each sentence
        is a 2-tuple of indices of words, indices of pos tags.
        Example of 2 sentences:
            [([1,5,2], [4,7,1]), ([1,2], [3,4])]
              w1 w2      p1 p2
              |------- s1 -------|

        w = word, p = pos tag, s = sentence

        This is as slow as the longest sentence - so bucketing sentences
        of same size can speed up training - prediction.
        """
        assert(len(inputs) >= 1)
        heads = kwargs.get('heads', None)
        labels = kwargs.get('labels', None)
        aux_labels = kwargs.get('aux_labels', None)
        swap = kwargs.get('swp', 0)

        calc_loss = ((heads is not None) and (labels is not None))
        # in order to process batches of different sized sentences using LSTM in chainer
        # we need to sort by sentence length.
        # The longest sentences in tokens need to be at the beginning of the
        # list, since chainer will simply not update the states corresponding
        # to the smallest sentences that have 'run out of tokens'.
        # We keep the permutation indices in order to reorder the output states,
        # since we want to map the activations to the inputs.
        perm_indices, sorted_batch = zip(*sorted(enumerate(zip(*inputs)),
                                                 key=lambda x: len(x[1][0]),
                                                 reverse=True))
        sorted_inputs = list(zip(*sorted_batch))
        extract_feat, extract_attn = flags
        extract_attn = False
        if extract_attn:
            print('Permutation indices:')
            print(perm_indices)

        self.loss = 0
        self.acc = 0
        aux_loss = 0
        aux_acc = 0
        if calc_loss:
            sorted_heads = [heads[i] for i in perm_indices]
            sorted_labels = [labels[i] for i in perm_indices]
            if self.beta > 0:
                sorted_aux_labels = [aux_labels[i] for i in perm_indices]
        else:
            sorted_heads, sorted_labels, sorted_aux_labels = None, None, None


        # get states from the RNN encoder
        rnn_states, embeddings = self.encoder(extract_feat, *sorted_inputs)
        final_states = rnn_states

        batch_stats = (self.encoder.batch_size,
                       self.encoder.max_seq_len,
                       self.encoder.col_lengths)
        batch_size, max_sent_len, col_lengths = batch_stats

        if calc_loss:
            # NOTE: We need the heads variables both in predict heads & labels
            # heads are seq_len - 1 in length because they don't include ROOT
            gold_heads = self.encoder.transpose_batch(sorted_heads)
            gold_tags = self.encoder.transpose_batch(sorted_labels)
            if self.beta > 0:
                gold_aux_tags = self.encoder.transpose_batch(sorted_aux_labels)
        else:
            gold_heads = None


        if self.beta > 0:
            aux_loss, aux_acc, pred_tags = self._predict_tags(final_states, self.encoder.mask, batch_stats,
                sorted_tags=gold_aux_tags)

        # check if we need to predict head/label
        # if swap == 1, we do MTL but want to do tagging first
        if self.alpha > 0 and swap <= 0:
            # perform subword attention
            if self.sub_attn:
                self.subword_embeds, _, sent_sub_lengths = self.encoder.attn_subword_embeds(self.units_dim, self.max_sub_len, *sorted_inputs)
                arcs, p_reps, sent_attn_vectors = self._predict_heads_attn(final_states, self.subword_embeds, self.encoder.mask, sent_sub_lengths, batch_stats,
                    extract_attn, sorted_heads=gold_heads)
            else:
                arcs = self._predict_heads(final_states, self.encoder.mask, batch_stats,
                    sorted_heads=gold_heads)

            # extract attention vector
            if extract_attn:
                u_arcs = arcs.data
                u_p_arcs = self.xp.argmax(u_arcs, axis=1)
                u_arc_preds = cuda.to_cpu(u_p_arcs)

            if self.debug or self.visualise:
                self.arcs = cuda.to_cpu(F.softmax(arcs).data)

            if self.treeify != 'none':
                # TODO: check multiple roots issue
                # We process the head scores to apply tree constraints
                arcs = cuda.to_cpu(arcs.data)
                # arcs are batch_size x sent_len + 1 x sent_len
                # axis 1 has the scores over the sentence
                # axis 2 is one shorter because we don't predict for root
                dd = DependencyDecoder()
                # sent length not taking root into account
                sent_lengths = [len(sent) for sent in sorted_inputs[0]]
                arc_preds = []
                if self.treeify == 'chu':
                    # Just remove cycles, non-projective trees are ok
                    for l, score_mat in zip(sent_lengths, arcs):
                        # remove fallout from batch size
                        trunc_score_mat = score_mat[:l+1, :l]
                        # DependencyDecoder expects a square matrix - fill root col with zeros
                        trunc_score_mat = np.pad(trunc_score_mat, ((0, 0), (1, 0)), 'constant')
                        nproj_arcs = dd.parse_nonproj(trunc_score_mat)[1:]
                        arc_preds.append(nproj_arcs)

                    # arc_preds = np.array([dd.parse_nonproj(each)[1:] for each in pd_arcs])
                elif self.treeify == 'eisner':
                    # Remove cycles and make sure trees are projective
                    for l, score_mat in zip(sent_lengths, arcs):
                        # remove fallout from batch size
                        trunc_score_mat = score_mat[:l+1, :l]
                        # DependencyDecoder expects a square matrix - fill root col with zeros
                        trunc_score_mat = np.pad(trunc_score_mat, ((0, 0), (1, 0)), 'constant')
                        proj_arcs = dd.parse_proj(trunc_score_mat)[1:]
                        arc_preds.append(proj_arcs)
                else:
                    raise ValueError('Unexpected method')
                p_arcs = self.encoder.transpose_batch(arc_preds, create_var=False)
            else:
                # We ignore tree constraints - head predictions may create cycles
                # we pass predict_labels the gpu object
                arcs = arcs.data
                p_arcs = self.xp.argmax(arcs, axis=1)
                arc_preds = cuda.to_cpu(p_arcs)
                p_arcs = np.swapaxes(p_arcs, 0, 1)


            if self.sub_attn:
                lbls = self._predict_labels_attn(final_states, p_arcs, p_reps, sent_attn_vectors, gold_heads,
                        batch_stats, extract_attn, sorted_labels=sorted_labels)
            else:
                lbls = self._predict_labels(final_states, p_arcs, gold_heads,
                        batch_stats, sorted_labels=sorted_labels)

            if self.debug or self.visualise:
                self.lbls = cuda.to_cpu(F.softmax(lbls).data)

            lbls = cuda.to_cpu(lbls.data)

            # we only bother actually getting the softmax values
            # if we are to visualise the results
            if self.visualise:
                # replace logits with prob from softmax - we pad with the exp
                # of the MIN_PAD - since that would have been the value if we passed
                # the MIN_PAD through the softmax
                # arcs = F.softmax(arcs)
                # arcs = F.pad(arcs, [(0, int(batch_size - self.num_active)), (0, 0)],
                #              'constant', constant_values=self.xp.exp(self.MIN_PAD))
                # lbls = F.softmax(lbls)
                # lbls = F.pad(lbls, [(0, int(batch_size - self.num_active)), (0, 0)],
                #              'constant', constant_values=self.xp.exp(self.MIN_PAD))
                self._visualise(self.arcs[0], self.lbls[0], sorted_heads[0], sorted_labels[0])
            # else:
                # arcs = F.pad(arcs, [(0, int(batch_size - self.num_active)), (0, 0)],
                #              'constant', constant_values=self.MIN_PAD)
                # lbls = F.pad(lbls, [(0, int(batch_size - self.num_active)), (0, 0)],
                #              'constant', constant_values=self.MIN_PAD)


        # normalize loss over all tokens seen
        total_tokens = np.sum(self.encoder.col_lengths)
        if self.beta > 0 and self.alpha > 0:
            if swap == 0:
                aux_loss = 0

        self.loss = ((self.alpha * self.loss) + (self.beta * aux_loss)) / total_tokens
        self.acc = aux_acc

        inv_perm_indices = [perm_indices.index(i) for i in range(len(perm_indices))]
        if self.debug or self.visualise:
            self.arcs = self.arcs[inv_perm_indices]
            self.lbls = self.lbls[inv_perm_indices]

        input_sent_lengths = [len(sent) for sent in inputs[0]]
        if self.beta > 0:
            final_tags = []
            for i in range(max_sent_len - 1):
                for j, tag_id in enumerate(pred_tags[i]):
                    if i == 0:
                        final_tags.append([tag_id.data])
                    else:
                        final_tags[j].append(tag_id.data)
            final_tags = [self.xp.array(f, dtype=self.xp.int32) for f in final_tags]
            tags = [final_tags[i] for i in inv_perm_indices]
            tag_preds = [tag_p[:l] for tag_p, l in zip(tags, input_sent_lengths)]
        else:
            tags = None
            tag_preds = None


        if self.alpha > 0 and swap <= 0:
            arcs = [arc_preds[i] for i in inv_perm_indices]
            lbls = lbls[inv_perm_indices]
            lbl_preds = np.argmax(lbls, axis=1)

            arc_preds = [arc_p[:l] for arc_p, l in zip(arcs, input_sent_lengths)]
            lbl_preds = [lbl_p[:l] for lbl_p, l in zip(lbl_preds, input_sent_lengths)]
        else:
            arc_preds, lbl_preds = None, None
        

        if extract_attn:
            print('Unlabeled head predictions:')
            u_arcs = [u_arc_preds[i] for i in inv_perm_indices]
            u_arc_preds = [arc_p[:l] for arc_p, l in zip(u_arcs, input_sent_lengths)]
            for u_pred in u_arc_preds:
                print(u_pred)

        if not extract_feat:
            return arc_preds, lbl_preds, tag_preds
        else:
            states = F.reshape(final_states, (max_sent_len, batch_size, -1))
            states = F.separate(states, axis=1)
            sorted_states = [states[i] for i in inv_perm_indices]
            
            # retrieve embedding layer
            embs = defaultdict(list)
            for i in range(len(embeddings)):
                for idx, vec in enumerate(embeddings[i]):
                    embs[idx].append(vec)

            # we skip START/END symbol
            sent_embs = list()
            for i in range(len(embs)):
                sent_embs.append(embs[i][1:-1])
            sorted_sent_embs = [sent_embs[i] for i in inv_perm_indices]

            return arc_preds, lbl_preds, tag_preds, sorted_states, sorted_sent_embs


    def _visualise(self, arcs, lbls, gold_heads, gold_labels):
        max_sent_len = len(arcs[1])
        one_hot_index = self.xp.zeros(max_sent_len, dtype=self.xp.float32)
        one_hot_arc = self.xp.zeros(max_sent_len+1, dtype=self.xp.float32)
        one_hot_lbl = self.xp.zeros(self.num_labels, dtype=self.xp.float32)
        for i in range(max_sent_len):
            correct_head_index = gold_heads[i]
            one_hot_arc[correct_head_index] = 1.
            one_hot_index[i] = 1.
            correct_lbl_index = gold_labels[i]
            one_hot_lbl[correct_lbl_index] = 1.
            six.print_(discrete_print('\n\nCur index : %-110s\nReal head : %-110s\nPred head : %-110s\n\n'
                                 'Real label: %-110s\nPred label: %-110s\n\n'
                                 'Sleep time: %.2f - change with up and down arrow keys') % (
                 '[%s] %d' % (bar(one_hot_index[:90]), i),
                 '[%s] %d |%d|' % (bar(one_hot_arc[:90]), correct_head_index, abs(correct_head_index - i)),
                    '[%s]' % bar(arcs[:, i].reshape(-1)[:90]),
                    '[%s] %s' % (bar(one_hot_lbl), UDepVocab.TAGS[correct_lbl_index]),
                    '[%s]' % bar(lbls[:, i].reshape(-1)),
                    self.sleep_time),
                  end='', flush=True)
            # reset values
            one_hot_index[i] = 0.
            one_hot_arc[correct_head_index] = 0.
            one_hot_lbl[correct_lbl_index] = 0.
            sleep(self.sleep_time)
