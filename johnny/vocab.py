import six
import pickle
from itertools import chain
from collections import Counter, namedtuple


# We reserve these indexes in all vocabs we create
# they aren't always needed but it simplifies our work
RESERVED = dict(START_SENTENCE=0,
                END_SENTENCE=1,
                ROOT=2,
                START_WORD=3,
                END_WORD=4,
                PAD=5)


reserved = namedtuple('Reserved', RESERVED.keys())(**RESERVED)

def augment_seq_nested(seq):
    # when we pad sentence that has been encoded on the subword level
    # we insert the index into a list so that it can be encoded
    # on the subword level - we don't want to have to deal with this
    # as a special case and encode 3 tokens on the word level!
    return tuple(chain(
            ([reserved.START_SENTENCE],),
            ([reserved.ROOT],),
            seq,
            ([reserved.END_SENTENCE],)))


def augment_seq(seq):
    return tuple(chain(
            (reserved.START_SENTENCE,),
            (reserved.ROOT,),
            seq,
            (reserved.END_SENTENCE,)))


def augment_word(seq):
    return tuple(chain(
            (reserved.START_WORD,),
            seq,
            (reserved.END_WORD,)))


class Vocab(object):
    """The tokens we know. Class defines a way to create the vocabulary
    and assign each known token to an index. All other tokens are replaced
    with the token UNK, which of course is UNK following the definition
    of Dr. UNK UNK from UNK.
    UNK is assigned the token 0 - because we like being arbitrary.
    The rest of the known tokens are sorted by frequency and assigned indices
    in such a manner.
    
    We keep the number of counts in order to be able to update our
    vocabulary later on. However, we throw away counts below or
    equal to threshold counts - because zipf's law and we don't
    have stocks in any companies producing ram chips.
    """

    special = dict(**RESERVED)
    special.update(UNK=len(special))
    reserved = namedtuple('Reserved', special.keys())(**special)

    def __init__(self, size=None, out_size=None, counts=None, threshold=0):
        """
            size: int - the number of tokens we can represent.
            We always represent UNK, START and END but we don't count
            them in len. Use out_size attribute for that.

            threshold: int - we throw away tokens with up to and including
            this many counts.
        """
        super(Vocab, self).__init__()
        if size is None:
            assert(out_size is not None)
            self.size = out_size - len(self.reserved)
            self.out_size = out_size
        elif out_size is None:
            assert(size is not None)
            self.out_size = size + len(self.reserved)
            self.size = size
        else:
            raise ValueError("Can't set both size and out_size")
        self.threshold = threshold
        self.index = None

    def __repr__(self):
        return ('Vocab object\ncapacity: %d\nactual size: %d\nthreshold: %d'
                % (self.size, len(self), self.threshold))

    def __len__(self):
        return len(self.index) + len(self.reserved)

    def __getitem__(self, key):
        return self.index[key]

    def _build_index(self):
        # we sort because in python 3 most_common is not guaranteed
        # to return the same order for elements with same count
        # when the code runs again. #fun_debugging
        candidates = sorted(self.counts.most_common(),
                            key=lambda x: (x[1], x[0]), reverse=True)
        limit = self.size
        offset = len(self.reserved)
        # we leave reserved indices to represent the UNK and the rest
        keep = candidates[:limit]
        if keep:
            keys, _ = zip(*keep)
            self.index = dict(zip(keys, range(offset, len(keys)+offset)))
        else:
            self.index = dict()

    def _threshold_counts(self):
        remove = []
        for key, c in six.iteritems(self.counts):
            if c <= self.threshold:
                remove.append(key)
        for key in remove:
            self.counts.pop(key)

    def fit(self, tokens):
        """Populate the vocabulary using the tokens as input.
        Tokens are expected to be a iterable of tokens."""
        self.counts = Counter(tokens)
        self._threshold_counts()
        self._build_index()
        return self

    def encode(self, tokens):
        """tokens: iterable of tokens to get indices for.
        Returns list of indices.  """
        return tuple(self.index.get(token, self.reserved.UNK) for token in tokens)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def save_txt(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in sorted(RESERVED.items(), key=lambda x: x[1]):
                f.write(str(value) + ' ' + key + '\n')
            f.write(str(len(RESERVED)) + ' ' + 'UNK' + '\n')
            for key, value in sorted(self.index.items(), key=lambda x: x[1]):
                f.write(str(value) + ' ' + key + '\n')

    @classmethod
    def load(cl, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class UPOSVocab(object):
    """ Universal dependencies part of speech tag vocabulary for version 2.
    Alphabetical listing

    ADJ: adjective
    ADP: adposition
    ADV: adverb
    AUX: auxiliary
    CCONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other
    """
    TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
            'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']

    def __init__(self):
        super(UPOSVocab, self).__init__()
        self.tags = list(RESERVED.keys())
        self.tags.extend(self.TAGS)
        self.index = dict((key, index) for index, key in enumerate(self.tags))

    def __repr__(self):
        return ('UPOSVocab object\nnum tags: %d\n' % len(self))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def fit(self):
        return self

    def encode(self, tags):
        """tags : iterable of tags """
        return tuple(self.index[tag] for tag in tags)

    def save_txt(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in sorted(self.index.items(), key=lambda x: x[1]):
                f.write(str(value) + ' ' + key + '\n')


class UDepVocab(object):
    """ Universal dependency relations label vocabulary.
    Alphabetical listing

    acl: clausal modifier of noun (adjectival clause)
    advcl: adverbial clause modifier
    advmod: adverbial modifier
    amod: adjectival modifier
    appos: appositional modifier
    aux: auxiliary
    case: case marking
    cc: coordinating conjunction
    ccomp: clausal complement
    clf: classifier
    compound: compound
    conj: conjunct
    cop: copula
    csubj: clausal subject
    dep: unspecified dependency
    det: determiner
    discourse: discourse element
    dislocated: dislocated elements
    expl: expletive
    fixed: fixed multiword expression
    flat: flat multiword expression
    goeswith: goes with
    iobj: indirect object
    list: list
    mark: marker
    nmod: nominal modifier
    nsubj: nominal subject
    nummod: numeric modifier
    obj: object
    obl: oblique nominal
    orphan: orphan
    parataxis: parataxis
    punct: punctuation
    reparandum: overridden disfluency
    root: root
    vocative: vocative
    xcomp: open clausal complement
    """
    TAGS = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
            'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj',
            'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat',
            'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod',
            'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root',
            'vocative', 'xcomp']

    def __init__(self):
        super(UDepVocab, self).__init__()
        self.tags = self.TAGS
        self.index = dict((key, index) for index, key in enumerate(self.tags))

    def __repr__(self):
        return ('UDepVocab object\nnum tags: %d' % len(self))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def fit(self, tokens):
        return self

    def encode(self, tags):
        """tags : iterable of tags """
        return tuple(self.index[tag] for tag in tags)

    def save_txt(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in sorted(self.index.items(), key=lambda x: x[1]):
                f.write(str(value) + ' ' + key + '\n')


class AuxVocab(object):
    """ Auxiliary label vocabulary.
    """
    def __init__(self, tags):
        super(AuxVocab, self).__init__()
        self.tags = tags
        self.index = dict((key, index) for index, key in enumerate(self.tags))

    def __repr__(self):
        return ('AuxVocab object\nnum labels: %d' % len(self))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def fit(self, tokens):
        return self

    def encode(self, tags):
        """tags : iterable of tags """
        return tuple(self.index[tag] for tag in tags)

    def save_txt(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for key, value in sorted(self.index.items(), key=lambda x: x[1]):
                f.write(str(value) + ' ' + key + '\n')


class MorphTags(object):
    """ 
    XFEATS inflectional features.
    Note that we only use inflectional tags from the universal dependencies v2.0.
    Here, we basically collect all possible universal tag, value pair from the training data
    and treat unknown tag, value pair during inference as an unknown tag.

    Universal features (15 inflectional features):
    Gender
    Animacy
    Number
    Case
    Definite
    Degree
    VerbForm
    Mood
    Tense
    Aspect
    Voice
    Evident
    Polarity
    Person
    Polite
    """
    IN_FEATS = ['Gender', 'Animacy', 'Number', 'Case', 'Definite',
                'Degree', 'VerbForm', 'Mood', 'Tense', 'Aspect',
                'Voice', 'Evident', 'Polarity', 'Person', 'Polite']

    def __init__(self):
        super(MorphTags, self).__init__()
        self.feats = self.IN_FEATS

    def __repr__(self):
        return ('XFEATS object\ninflectional tags: %d\n' % len(self))

    def __len__(self):
        return len(self.feats)

    def get_tags(self):
        return self.feats


class AbstractVocab(object):
    """Used when we don't know what labels to expect"""

    def __init__(self, with_reserved=True, mutable=True):
        super(AbstractVocab, self).__init__()
        self.index = dict(**RESERVED) if with_reserved else dict()
        self.rev_index = dict((val, key) for key, val in RESERVED.items()) \
                         if with_reserved else dict()
        self.mutable = mutable

    def __repr__(self):
        return ('AbstractVocab object\nnum tags: %d' % (len(self)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return self.index[key]

    def fit(self, tokens):
        return self

    def encode(self, tags):
        """tags : iterable of tags """
        l = []
        for tag in tags:
            if self.mutable:
                if tag not in self.index:
                    new_idx = len(self.index)
                    self.index[tag] = new_idx
                    self.rev_index[new_idx] = tag
            l.append(self.index[tag])
        return tuple(l)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cl, filepath):
        with open(filepath, 'rb') as f:
            v = pickle.load(f)
            v.mutable = False
            return v

