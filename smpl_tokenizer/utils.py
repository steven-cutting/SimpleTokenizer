__title__ = 'smpl_tokenizer'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '02/06/2016'
__copyright__ = "smpl_tokenizer Copyright (C) 2015  Steven Cutting"

import json
import os
import re

import toolz as tlz
filter_c = tlz.curry(tlz.filter)
map_c = tlz.curry(tlz.map)


with open(os.path.join(os.path.dirname(__file__), 'stopwords.json')) as fp:
        STOPWORDS = frozenset(json.load(fp)['stopwords'])


# ---------------------------


@tlz.curry
def join_strings(glue, strings):
    """
    Same as:
        In [0]: glue = "-"
        In [1]: strings = ["foo", "bar"]
        In [2]: glue.join(strings)
        Out[2]: "foo-bar"
    """
    return glue.join(strings)


# ---------------------------


@tlz.curry
def split_on_reg(regpattern, string):
    """
    Splits the string based on the regex pattern 'regpattern'.
    """
    return (bit for bit in re.split(regpattern, string))


def splitter_of_words(string):
    """
    Splits 'string' on non alphanumeric characters.
    """

    non_alpha_num = r"""[,-/\#\$\%\(\)\*\+\<\=\>\@\[\\\]\^\`\{\|\}\~ :?&;!'"\n\t\r]+"""
    return split_on_reg(non_alpha_num, string)


def lower(string):
    """
    Makes string all lowercase.
    """
    return str.lower(string)


# ---------------------------


def filter_whitespace(tokenset):
    """
    Filters out tokens that are only whitespace.
    """
    return tlz.filter(tlz.compose(bool, str.strip), tokenset)


@tlz.curry
def filter_shorter_than(n, tokenset):
    """
    Filters out tokens that have less than 'n' characters.
    """
    return tlz.filter(lambda tkn: len(tkn) >= n, tokenset)


@tlz.curry
def filter_longer_than(n, tokenset):
    """
    Filters out tokens that have 'n' characters or more.
    """
    return tlz.filter(lambda tkn: len(tkn) < n, tokenset)


def filter_stopwords(tokenset):
    """
    Filters out stopwords.
    """
    return tlz.filter(lambda tkn: tkn not in STOPWORDS, tokenset)


# ---------------------------


def freq(tokenset):
    """
    Find number of occurrences of each value 'tokenset'.
    """
    return tlz.pipe(tokenset,
                    tlz.frequencies,
                    dict.items)
