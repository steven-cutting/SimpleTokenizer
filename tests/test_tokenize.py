__title__ = 'smpl_tokenizer'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '02/06/2016'
__copyright__ = "smpl_tokenizer Copyright (C) 2015  Steven Cutting"


import pytest
import toolz as tlz
reduce_c = tlz.curry(tlz.reduce)

from smpl_tokenizer import tokenize as tkn


@pytest.mark.parametrize("n,string,expected",
                         [(1, "foo-bar", [("foo", ), ("bar", )]),
                          (1, "foobazbar", [("foobazbar", )]),
                          (1, "foo*bar*baz", [("foo", ), ("bar", ), ("baz", )]),
                          (2, "foo*bar*baz", [("foo", "bar"), ("bar", "baz")]),
                          ])
def test__ngram_tuples(n, string, expected):
    assert(list(tkn.ngram_tuples(n, string)) == expected)


@pytest.mark.parametrize("n,string,expected",
                         [(1, "foo-bar", ["foo", "bar"]),
                          (2, "foo-bar", ["foo_bar"]),
                          (3, "foo-bar", []),
                          (1, "foobazbar", ["foobazbar"]),
                          (2, "foobazbar", []),
                          (3, "foobazbar", []),
                          (1, "foo*bar*baz", ["foo", "bar", "baz"]),
                          (2, "foo*bar*baz", ["foo_bar", "bar_baz"]),
                          (3, "foo*bar*baz", ["foo_bar_baz"]),
                          ])
def test__ngram(n, string, expected):
    assert(list(tkn.ngram(n, string)) == expected)


@pytest.mark.parametrize("string,expected",
                         [("foo-bar", ["foo", "bar"]),
                          ("foobazbar", ["foobazbar"]),
                          ("foo*bar*baz", ["foo", "bar", "baz"]),
                          ])
def test__uni_gram(string, expected):
    assert(list(tkn.uni_gram(string)) == expected)


@pytest.mark.parametrize("string,expected",
                         [("foo-bar", ["foo_bar"]),
                          ("foobazbar", []),
                          ("foo*bar*baz", ["foo_bar", "bar_baz"]),
                          ])
def test__bi_gram(string, expected):
    assert(list(tkn.bi_gram(string)) == expected)


@pytest.mark.parametrize("string,expected",
                         [("foo-bar", []),
                          ("foobazbar", []),
                          ("foo*bar*baz", ["foo_bar_baz"]),
                          ])
def test__tri_gram(string, expected):
    assert(list(tkn.tri_gram(string)) == expected)


sum_tally_tuples = lambda tpls: reduce_c(lambda x, y: x+y[1], tpls, 0)
extext = tlz.reduce(lambda x, y: x+y, ["aaa " * 20,
                                       "bbb " * 10,
                                       "ccc " * 3,
                                       "ddd " * 1])


@pytest.mark.parametrize("string,length,total,parser",
                         [(extext, 4, 34, tkn.uni_gram),

                          ])
def test___gram_counts(string, length, total, parser):
    bow = tkn.gram_counts(parser, string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)


@pytest.mark.parametrize("string,length,total",
                         [(extext, 4, 34),

                          ])
def test__uni_gram_counts(string, length, total):
    bow = tkn.uni_gram_counts(string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)


@pytest.mark.parametrize("string,length,total",
                         [(extext, 6, 33),

                          ])
def test__bi_gram_counts(string, length, total):
    bow = tkn.bi_gram_counts(string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)


@pytest.mark.parametrize("string,length,total",
                         [(extext, 8, 32),

                          ])
def test__tri_gram_counts(string, length, total):
    bow = tkn.tri_gram_counts(string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)


@pytest.mark.parametrize("wordcounts,exbow,exdict",
                         [([("aaa", 20),
                            ("bbb", 10),
                            ("ccc", 3),
                            ("ddd", 1)],
                           [(0, 20),
                            (1, 10),
                            (2, 3),
                            (3, 1)],
                           {0: "aaa", 1: "bbb",
                            2: "ccc", 3: "ddd"}),
                          ])
def test__bag_of_words(wordcounts, exbow, exdict):
    bow, dict_ = tkn.bag_of_words(wordcounts)
    assert(bow == exbow)
    assert(dict_ == exdict)


@pytest.mark.parametrize("string,excounts,parser",
                         [(extext,
                           [("aaa", 20),
                            ("bbb", 10),
                            ("ccc", 3),
                            ("ddd", 1)],
                           tkn.uni_gram),
                          (extext,
                           [("aaa_aaa", 19),
                            ("aaa_bbb", 1),
                            ("bbb_bbb", 9),
                            ("bbb_ccc", 1),
                            ("ccc_ccc", 2),
                            ("ccc_ddd", 1)],
                           tkn.bi_gram),
                          ])
def test__text_to_bow(string, excounts, parser):
    t2bow = tkn.text_to_bow(parser)  # testing ability to curry
    bow, dict_ = t2bow(string)
    gram2count = sorted([(dict_[i], c) for i, c in bow], key=tlz.first)
    assert(gram2count == excounts)


@pytest.mark.parametrize("string,excounts",
                         [(extext,
                           [("aaa", 20),
                            ("bbb", 10),
                            ("ccc", 3),
                            ("ddd", 1)]),
                          ])
def test__text_to_uni_bow(string, excounts):
    bow, dict_ = tkn.text_to_uni_bow(string)
    gram2count = sorted([(dict_[i], c) for i, c in bow], key=tlz.first)
    assert(gram2count == excounts)


@pytest.mark.parametrize("string,excounts",
                         [(extext,
                           [("aaa_aaa", 19),
                            ("aaa_bbb", 1),
                            ("bbb_bbb", 9),
                            ("bbb_ccc", 1),
                            ("ccc_ccc", 2),
                            ("ccc_ddd", 1)]),
                          ])
def test__text_to_bi_bow(string, excounts):
    bow, dict_ = tkn.text_to_bi_bow(string)
    gram2count = sorted([(dict_[i], c) for i, c in bow], key=tlz.first)
    assert(gram2count == excounts)
