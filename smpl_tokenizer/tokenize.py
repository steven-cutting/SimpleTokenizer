__title__ = 'smpl_tokenizer'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '02/06/2016'
__copyright__ = "smpl_tokenizer Copyright (C) 2015  Steven Cutting"

from itertools import repeat, count, izip

import toolz as tlz
sliding_window_c = tlz.curry(tlz.sliding_window)
map_c = tlz.curry(tlz.map)

from smpl_tokenizer import utils


@tlz.curry
def ngram_tuples(n, string, minlen=3, maxlen=25):
    return tlz.pipe(string,
                    utils.lower,
                    utils.splitter_of_words,
                    utils.filter_whitespace,
                    utils.filter_shorter_than(minlen),
                    utils.filter_longer_than(maxlen),
                    sliding_window_c(n))


@tlz.curry
def ngram(n, string, minlen=3, maxlen=25):
    return tlz.pipe(string,
                    ngram_tuples(n, minlen=minlen, maxlen=maxlen),
                    map_c(utils.join_strings("_")))


uni_gram = ngram(1)
bi_gram = ngram(2)
tri_gram = ngram(3)


@tlz.curry
def gram_counts(parser, string):
    return tlz.pipe(string,
                    parser,
                    utils.freq)


uni_gram_counts = gram_counts(uni_gram)
bi_gram_counts = gram_counts(bi_gram)
tri_gram_counts = gram_counts(tri_gram)


def bag_of_words(gramcounts):
    dictionary = dict()
    bow = []
    for i,gram in enumerate(gramcounts):
        dictionary[i] = gram[0]
        bow.append((i, gram[1]))
    return bow, dictionary


@tlz.curry
def text_to_bow(parser, string):
    """
    Transforms 'string' into a bag of words representation.
    It uses the supplied 'parser' to parse the string.
    """
    return tlz.pipe(string,
                    parser,
                    utils.freq,
                    bag_of_words)


text_to_uni_bow = text_to_bow(uni_gram)
text_to_bi_bow = text_to_bow(bi_gram)
