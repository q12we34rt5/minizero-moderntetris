#!/usr/bin/env python3
"""Summarize clairvoyance-gap eval results.

Each results/*.txt line is `SelfPlay <bool> <len> <len> <total_reward>`.
A game shorter than max_episode_steps ended in death.
"""
import glob
import math
import os
import sys

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
here = os.path.dirname(os.path.abspath(__file__))


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return n, m, sd, sd / math.sqrt(n)


rows = {}
for path in sorted(glob.glob(os.path.join(here, 'results', '*.txt'))):
    games = [line.split() for line in open(path) if line.startswith('SelfPlay')]
    if not games:
        continue
    lengths = [int(g[2]) for g in games]
    scores = [float(g[4]) for g in games]
    tag = os.path.basename(path)[:-4]
    rows[tag] = (scores, lengths)
    n, m, sd, se = stats(scores)
    deaths = sum(1 for length in lengths if length < MAX_STEPS)
    _, ml, _, _ = stats(lengths)
    print(f'{tag}\n  games={n}  score={m:.2f} ± {se:.2f} (sd {sd:.2f})  '
          f'death_rate={deaths / n:.1%}  mean_len={ml:.1f}')

base = [t for t in rows if t.endswith('_true-future')]
for t in base:
    prefix = t[:-len('true-future')]
    _, ma, _, sa = stats(rows[t][0])
    print(f'\ngap vs {t}:')
    for other in rows:
        if other == t or not other.startswith(prefix):
            continue
        _, mb, _, sb = stats(rows[other][0])
        gap, se = ma - mb, math.sqrt(sa ** 2 + sb ** 2)
        print(f'  {other[len(prefix):]:<20} {gap:7.2f} ± {se:.2f}  ({gap / ma:.1%} of true-future)')
