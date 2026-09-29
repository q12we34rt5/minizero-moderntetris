#!/usr/bin/env python3
"""How well does the search's value match what the game actually paid out?

Every self-play record stores, per move, the MCTS root value V[] (the search's
estimate) and the reward R[] the env gave. Discounting the rewards that actually
followed gives the realized return G, so V - G is what the search was wrong by.

This separates two things that matter for the clairvoyant-vs-honest comparison:

  bias  mean(V - G) -- a search that can see the real future computes E[max]
        rather than max E, so its value should read high against what a policy
        that cannot see the future actually collects.
  rmse  spread of V - G -- how noisy the value the learner trains on is.

Truncation: G is summed to the end of the record, so moves near the end miss
their tail. Only moves with at least --min-tail remaining are counted (the
default drops less than gamma^60 ~ 6% of a step's weight).

Policy entropy over the visit distribution P[] is reported alongside, as a
rough read on how sharp the policy target is.

Usage:
  sgf_value_calibration.py [--discount D] [--min-tail N] <sgf_file>...
"""
import argparse
import math
import re
import sys

MOVE_RE = re.compile(r';B\[(\d+)\]P\[([^\]]*)\]V\[([-0-9.e+]+)\]R\[([-0-9.e+]+)\]')


def entropy(policy_field):
    total = 0.0
    probs = []
    for part in policy_field.split(','):
        if ':' not in part:
            continue
        p = float(part.split(':')[1])
        if p > 0:
            probs.append(p)
            total += p
    if total <= 0:
        return None
    h = 0.0
    for p in probs:
        q = p / total
        h -= q * math.log(q)
    return h


def analyze(path, discount, min_tail):
    errors, values, returns, entropies = [], [], [], []
    games = 0
    for line in open(path, errors='replace'):
        if not line.startswith('(;GM'):
            continue
        games += 1
        moves = [(float(m.group(3)), float(m.group(4)), m.group(2))
                 for m in MOVE_RE.finditer(line)]
        if not moves:
            continue
        # realized discounted return from each move, computed backwards
        tail = 0.0
        g = [0.0] * len(moves)
        for t in range(len(moves) - 1, -1, -1):
            tail = moves[t][1] + discount * tail
            g[t] = tail
        for t, (v, _, policy) in enumerate(moves):
            if len(moves) - t < min_tail:
                continue
            values.append(v)
            returns.append(g[t])
            errors.append(v - g[t])
            h = entropy(policy)
            if h is not None:
                entropies.append(h)
    return games, values, returns, errors, entropies


def stats(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--discount', type=float, default=0.954993)
    ap.add_argument('--min-tail', type=int, default=60)
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    for path in args.files:
        games, values, returns, errors, entropies = analyze(path, args.discount, args.min_tail)
        if not errors:
            print(f'{path}: no usable moves')
            continue
        mv, sv = stats(values)
        mg, sg = stats(returns)
        me, se = stats(errors)
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        mh, _ = stats(entropies)
        # correlation between the search's value and what actually happened
        cov = sum((v - mv) * (g - mg) for v, g in zip(values, returns)) / len(values)
        corr = cov / (sv * sg) if sv > 0 and sg > 0 else float('nan')
        print(f'{path}')
        print(f'  games {games}, moves used {len(errors)}')
        print(f'  search value V: {mv:7.2f} ± {sv:5.2f}   realized return G: {mg:7.2f} ± {sg:5.2f}')
        print(f'  bias  mean(V-G) {me:+7.2f} ± {se:5.2f}   rmse {rmse:6.2f}   corr(V,G) {corr:+.3f}')
        print(f'  policy entropy {mh:.3f} nats')
    return 0


if __name__ == '__main__':
    sys.exit(main())
