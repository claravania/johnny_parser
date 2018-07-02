import chainer
import pickle
import pdb
import os
from tqdm import tqdm
from johnny.dep import UDepLoader
from johnny.metrics import Average, UAS, LAS
from train import visualise_dict, data_to_rows, to_batches
from mlconf import ArgumentParser, Blueprint


def test_loop(args, bp, test_set, feat_file=None, label_file=None):

    extract_attn = args.extract_attn
    extract_feat = args.extract_feat
    output_tags = args.output_tags

    model_path = bp.model_path
    vocab_path = bp.vocab_path

    with open(vocab_path, 'rb') as pf:
        vocabs = pickle.load(pf)

    (v_word, v_pos, v_arcs, v_aux) = vocabs

    if extract_attn:
        v_word.save_txt('word_vocab_attn.' + bp.dataset.lang)

    test_rows = data_to_rows(test_set, vocabs, bp)

    v_arcs_rev_index = dict((val, key) for key, val in v_arcs.index.items())
    if v_aux:
        v_aux_rev_index = dict((val, key) for key, val in v_aux.index.items())

    # Remove all info we are going to predict
    # to make sure we don't make a fool of ourselves
    # if we have a bug and gold data stays in its place
    test_set.unset_heads()
    test_set.unset_labels()

    built_bp = bp.build()
    model = built_bp.model
    
    if bp.gpu_id >= 0:
        model.to_gpu(bp.gpu_id)
    chainer.serializers.load_npz(model_path, model)

    # test
    tf_str = ('Eval - test : batch_size={0:d}, mean loss={1:.2f}, '
              'mean UAS={2:.3f} mean LAS={3:.3f}')
    with tqdm(total=len(test_set)) as pbar, \
        chainer.using_config('train', False), \
        chainer.no_backprop_mode():

        mean_loss = Average()
        mean_acc = Average()
        u_scorer = UAS()
        l_scorer = LAS()
        index = 0

        # NOTE: IMPORTANT!!
        # BATCH SIZE is important here to reproduce the results
        # for the cnn - since changing the batch size changes
        # has the effect of different words having different padding.
        # NOTE: test_mean_loss changes because it is averaged
        # across batches, so changing the number of batches affects it

        BATCH_SIZE = 256

        if extract_feat:
            flabel = open(label_file, 'w', encoding='utf-8')
            ffeat = open(feat_file, 'w', encoding='utf-8')


        batch_id = 0
        idx_sample = 0
        num_tokens = 0
        mtl = False
        for batch in to_batches(test_rows, BATCH_SIZE, sort=False):

            batch_size = 0
            batch_id += 1
            if len(batch) == 5:
                mtl = True
                word_batch, pos_batch, head_batch, label_batch, aux_label_batch = zip(*batch)
            else:
                word_batch, pos_batch, head_batch, label_batch = zip(*batch)
                aux_label_batch = None

            if extract_attn:
                print('Batch:', batch_id)

            flags = [extract_feat, extract_attn]
            if UNLABELLED:
                arc_preds, lbl_preds = model(flags, word_batch, pos_batch,
                                             heads=None, labels=None)
            else:
                if not extract_feat:
                    arc_preds, lbl_preds, tag_preds = model(flags, word_batch, pos_batch,
                                         heads=head_batch, labels=label_batch, aux_labels=aux_label_batch)
                else:
                    arc_preds, lbl_preds, tag_preds, states, embs = model(flags, word_batch, pos_batch,
                                         heads=head_batch, labels=label_batch, aux_labels=aux_label_batch)

            if extract_feat:
                true_bs = len(word_batch)
                for sent_idx in range(idx_sample, idx_sample + true_bs):
                    words = test_set.words[sent_idx]
                    upostags = test_set.upostags[sent_idx]
                    xfeats = test_set.feats[sent_idx]
                    if extract_feat == 'enc':
                        state = states[sent_idx - idx_sample]
                    elif extract_feat == 'emb':
                        state = embs[sent_idx - idx_sample]
                    else:
                        sys.exit('Wrong mode, please select enc or emb.')

                    assert len(words) == len(upostags) == len(xfeats)

                    for idx in range(len(words)):
                        labels = [idx + 1, words[idx], upostags[idx], xfeats[idx]]
                        labels = '\t'.join([str(l) for l in labels])
                        flabel.write(labels + '\n')

                        feats = ','.join([str(x) for x in state[idx + 1].data])
                        ffeat.write(feats + '\n')

                        num_tokens += 1
                        
                    flabel.flush()
                    ffeat.flush()
                idx_sample += true_bs

            if extract_attn:
                print('Data:')
                for it in range(len(word_batch)):
                    print(word_batch[it], ' ||| ', pos_batch[it], '|||', tuple(arc_preds[it]), '|||', 
                        tuple(lbl_preds[it]), ' ||| ', head_batch[it], ' ||| ', label_batch[it])
                print()

            loss = model.loss
            loss_value = float(loss.data)

            if mtl:
                acc = model.acc
                acc_value = float(acc.data)
                mean_acc(acc_value)
                tag_acc = mean_acc.score
            else:
                tag_acc = 0.0

            if arc_preds and lbl_preds:
                for p_arcs, p_lbls, t_arcs, t_lbls in zip(arc_preds, lbl_preds, head_batch, label_batch):
                    u_scorer(arcs=(p_arcs, t_arcs))
                    l_scorer(arcs=(p_arcs, t_arcs), labels=(p_lbls, t_lbls))

                    test_set[index].set_heads(p_arcs)
                    str_labels = [v_arcs_rev_index[l] for l in p_lbls]
                    test_set[index].set_labels(str_labels)

                    index += 1
                    batch_size += 1
            else:
                for tags, t_arcs, t_lbls in zip(tag_preds, head_batch, label_batch):
                    test_set[index].set_heads(t_arcs)
                    str_labels = [v_arcs_rev_index[l] for l in t_lbls]
                    test_set[index].set_labels(str_labels)

                    tags = tuple(tags.tolist())
                    str_tags = [v_aux_rev_index[l] for l in tags]
                    test_set[index].set_feats(str_tags)

                    index += 1
                    batch_size += 1

            mean_loss(loss_value)
            out_str = tf_str.format(batch_size, mean_loss.score, u_scorer.score, l_scorer.score, tag_acc)
            pbar.set_description(out_str)
            pbar.update(batch_size)

        if extract_feat:
            flabel.close()
            ffeat.close()
            print('Total token vectors:', num_tokens)

    # make sure you aren't a dodo
    assert(index == len(test_set))

    stats = {'test_mean_loss': mean_loss.score,
             'test_uas': u_scorer.score,
             'test_las': l_scorer.score,
             'aux_acc': tag_acc}

    # TODO: save these
    bp.test_results = stats
    for key, val in sorted(stats.items()):
        print('%s: %s' % (key, val))


if __name__ == "__main__":
    # needed to import train to visualise_train
    parser = ArgumentParser(description='Dependency parser trainer')
    parser.add_argument('--blueprint', required=True, type=str,
                        help='Path to .bp blueprint file produces by training.')
    parser.add_argument('--test_file', required=True, type=str,
                        help='Conll file to use for testing')
    parser.add_argument('--conll_out', type=str, default='out.conllu',
                        help='If specified writes conll output to this file')
    parser.add_argument('--unlabelled', action='store_true',
                        help='whether we are passing labels or not')
    parser.add_argument('--treeify', type=str, default='chu',
                        help='algorithm to postprocess arcs with. '
                        'Choose chu to allow for non projectivity, else eisner')
    parser.add_argument('--extract_feat', type=str, default=None,
                        help='Whether to extract features or not.')
    parser.add_argument('--extract_attn', type=str, default=None,
                        help='Whether to extract attention vectors (for attention model only).')
    parser.add_argument('--output_tags', action='store_true',
                        help='Whether to output tag predictions.')
    parser.add_argument('--lang', type=str, default='english',
                        help='Language (optional, only for extracting feature). ')

    args = parser.parse_args()

    CONLL_OUT = args.conll_out
    UNLABELLED = args.unlabelled
    TREEIFY = args.treeify
    extract_feat = args.extract_feat
    extract_attn = args.extract_attn
    lang = args.lang

    blueprint = Blueprint.from_file(args.blueprint)
    blueprint.model.treeify = TREEIFY
    feat_file = None
    label_file = None

    test_data = UDepLoader.load_conllu(args.test_file)

    filename = os.path.basename(args.test_file).replace('.conllu', '')

    if extract_feat:
        path = os.path.join('features', lang)
        if not os.path.exists(path):
            os.makedirs(path)

        feat_file = os.path.join(path, filename + '.feat')
        label_file = os.path.join(path, filename + '.lbl')

    test_loop(args, blueprint, test_data, feat_file=feat_file, label_file=label_file)

    if CONLL_OUT:
        # test_data.save(blueprint.model_path.replace('.model', '.conllu'))
        test_data.save(os.path.join('outputs', CONLL_OUT))
