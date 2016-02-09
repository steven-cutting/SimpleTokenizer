__title__ = 'smpl_tokenizer'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '02/06/2016'
__copyright__ = "smpl_tokenizer Copyright (C) 2015  Steven Cutting"


from itertools import repeat
from functools import partial
import types

import pytest
import toolz as tlz
map_c = tlz.curry(tlz.map)
reduce_c = tlz.curry(tlz.reduce)

from smpl_tokenizer import utils


is_generator = lambda obj: isinstance(obj, types.GeneratorType)

var_len_strings = lambda n: list(tlz.take(n, tlz.iterate(lambda string: string + "a", "")))


"""
@pytest.mark.parametrize("test_input,expected", [
        ("3+5", 8),
        ("2+4", 6),
        ("6*9", 42),
])
def _eval(test_input, expected):
        assert _eval(test_input) == expected


@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_foo(x, y):
        pass
"""


@pytest.mark.parametrize("glue,strings,expected",
                         [("-", ["foo", "bar"], "foo-bar"),
                          ("baz", ["foo", "bar"], "foobazbar"),
                          ("*", ["foo", "bar", "baz"], "foo*bar*baz"),
                          ])
def test__join_strings(glue, strings, expected):
    assert(utils.join_strings(glue, strings) == expected)


@pytest.mark.parametrize("string",
                         ["monkey",
                          "monkey--monkey",
                          ])
def test__split_on_reg__0(string):
    splitter = utils.split_on_reg(r"[-]+")
    assert(is_generator(splitter(string)))


@pytest.mark.parametrize("string,pattern,length",
                         [("monkey", r"[-]+", 1),
                          ("monkey--monkey", r"[-]+", 2),
                          ("monkey--monkey---monkey", r"[-]+", 3),
                          ])
def test__split_on_reg__1(string, pattern, length):
    tokens = list(utils.split_on_reg(pattern, string))
    assert(len(tokens) == length)
    assert(len(set(tokens)) == 1)


@pytest.mark.parametrize("string",
                         ["monkey",
                          "monkey--monkey",
                          ])
def test__splitter_of_words__0(string):
    assert(is_generator(utils.splitter_of_words(string)))


@pytest.mark.parametrize("string,length,unique",
                         [("monkey", 1, 1),
                          ("monkey--monkey", 2, 1),
                          ("monkey--monkey---monkey", 3, 1),
                          ("monkey&&monkey", 2, 1),
                          ("foo foo", 2, 1),
                          ("bar  bar", 2, 1),
                          ("monkey-/#$%() \\*+=?.:;[]!&<>@monkey", 2, 1),
                          ("foo-/#$%() \\bar*+=?.:;[]!&<>@baz", 3, 3),
                          ("foo-/bar#$%()foo \\bar*+=?.:baz;[]!foo&<>@baz", 7, 3),
                          ("he's", 2, 2),
                          ])
def test__splitter_of_words__1(string, length, unique):
    tokens = list(utils.splitter_of_words(string))
    assert(len(tokens) == length)
    assert(len(set(tokens)) == unique)


@pytest.mark.parametrize("string,expected",
                         [("MONKEY", "monkey"),
                          ("Foo Bar baz", "foo bar baz"),
                          ])
def test__lower(string, expected):
    assert(utils.lower(string) == expected)


@pytest.mark.parametrize("tokenset,count",
                         [([" ", "", "\t", "\n", "\r"], 0),
                          ([" \t\n\r"], 0),
                          (["MONKEY", "monkey", "foo bar baz"], 3),
                          (["monkey\t", "foo\n", "bar\r"], 3),
                          ])
def test__filter_whitespace(tokenset, count):
    assert(len(list(utils.filter_whitespace(tokenset))) == count)


@pytest.mark.parametrize("tokenset,mintokenlen,count",
                         [(var_len_strings(20), 0, 20),
                          (var_len_strings(20), 5, 15),
                          (var_len_strings(20), 20, 0),
                          ])
def test__filter_shorter_than(tokenset, mintokenlen, count):
    length = tlz.pipe(tokenset,
                      utils.filter_shorter_than(mintokenlen),
                      list,
                      len)
    assert(length == count)


@pytest.mark.parametrize("tokenset,mintokenlen,count",
                         [(var_len_strings(20), 0, 0),
                          (var_len_strings(20), 5, 5),
                          (var_len_strings(20), 20, 20),
                          ])
def test__filter_longer_than(tokenset, mintokenlen, count):
    length = tlz.pipe(tokenset,
                      utils.filter_longer_than(mintokenlen),
                      list,
                      len)
    assert(length == count)


sum_tally_tuples = lambda tpls: reduce_c(lambda x, y: x+y[1], tpls, 0)


@pytest.mark.parametrize("tokenset,length,total",
                         [(tlz.concatv(["a"] * 20,
                                       ["b"] * 10,
                                       ["c"] * 3,
                                       ["d"] * 1,
                                       ), 4, 34),

                          ])
def test__freq(tokenset, length, total):
    bow = utils.freq(tokenset)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)
