from random import shuffle
from itertools import islice

INF = float('inf')


def irange(start, stop=None, step=1):  # np.arange
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield start
        start += step


def argmin(function, sequence):
    values = list(sequence)
    scores = [function(x) for x in values]
    return values[scores.index(min(scores))]


def pairs(lst):
    return zip(lst[:-1], lst[1:])


def merge_dicts(*args):
    result = {}
    for d in args:
        result.update(d)
    return result
    # return dict(reduce(operator.add, [d.items() for d in args]))


def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)


def randomize(sequence):
    shuffle(sequence)
    return sequence


def take(iterable, n=INF):
    if n == INF:
        n = None  # NOTE - islice takes None instead of INF
    elif n == None:
        n = 0  # NOTE - for some of the uses
    return islice(iterable, n)


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    enums['names'] = sorted(enums.keys(), key=lambda k: enums[k])
    return type('Enum', (), enums)
