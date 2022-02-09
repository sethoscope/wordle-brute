#!/usr/bin/env python3
#
# Deduce the target word just from the shares of other people's games.
# (The shares show the color responses for each guess, without showing
# the letters.)


import sys
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
import os
from collections import defaultdict
import pickle

class Response():
    ABSENT = 0
    PRESENT = 1
    CORRECT = 2

    SQUARES = [chr(11036),    # grey square, light theme
               chr(129000),   # yellow square
               chr(129001)]   # green square

    DARK_THEME_ABSENT = chr(11035)
    LIGHT_THEME_ABSENT = chr(11036)

    def __init__(self, target, guess):
        self.info = self.make_response(target, guess)

    @classmethod
    def canonicalize_blocks(cls, s):
        s = s.strip()
        return s.strip().replace(cls.DARK_THEME_ABSENT,
                                 cls.LIGHT_THEME_ABSENT)

    def __hash__(self):
        return hash(tuple(self.info))

    def __eq__(self, other):
        return self.info == other.info

    def __str__(self):
        return ''.join(self.SQUARES[c] for c in self.info)


    # translated from the wordle code, so it should give identical results
    @classmethod
    def make_response(cls, target, guess):
        '''Returns list of ints corresponding to ABSENT, PRESENT, CORRECT'''
        assert(len(target) == len(guess))
        guess_avail = [True] * len(guess)
        target_avail = [True] * len(target)
        result = [cls.ABSENT] * len(guess)
        
        for i in range(len(guess)):
            if guess[i] == target[i] and target_avail[i]:
                result[i] = cls.CORRECT
                guess_avail[i] = False
                target_avail[i] = False

        for i in range(len(guess)):
            if guess_avail[i]:
                for j in range(len(target)):
                    if target_avail[j] and (guess[i] == target[j]):
                        result[i] = cls.PRESENT
                        target_avail[j] = False;
        return tuple(result)


class Table():
    def __init__(self, data):
        self.data = data
    
    @classmethod
    def make_table(cls, targetwords, guesswords):
        '''
        Returns a dict of str : set, where str is a string of colored blocks
        and set is a set of target words for which some legal guess would
        produce that pattern of blocks.
        '''
        logging.info('Making lookup table. This will take a while.')
        data = defaultdict(set)
        for t in targetwords:
            logging.debug(f'Starting {t}')
            for g in guesswords:
                if g != t:    # skip the trivial correct guess
                    bs = str(Response(t, g))
                    data[bs].add(t)
                    #logging.debug(f'Adding {t} to {bs} due to {g}: {data[bs]}')
        return Table(data)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            logging.debug('Loading lookup table.')
            return Table(pickle.load(f))

    def save(self, filename):
        with open(filename, 'wb') as f:
            logging.debug('Saving lookup table.')
            pickle.dump(self.data, f)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


def main():
    description = ''
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--table', help='file containing lookup table data')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('target_words', type=FileType('r'),
                        help='words the computer might have chosen')
    parser.add_argument('guess_words', type=FileType('r'),
                        help='words people are allowed to guess')
    parser.add_argument('shares', type=FileType('r'),
                        help='shared games',
                        default=sys.stdin, nargs="?")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)


    target_words = [line.strip() for line in args.target_words.readlines()]
    guess_words = [line.strip() for line in args.guess_words.readlines()]

    if args.table and os.path.exists(args.table):
        table = Table.load(args.table)
    else:
        table = Table.make_table(target_words, guess_words)
        if args.table:
            table.save(args.table)

    responses = set(Response.canonicalize_blocks(row)
                    for row in args.shares.readlines())
    candidates = set(target_words)
    logging.info(f'Starting with {len(candidates)} possible solutions.')
    for r in responses:
        if r in table:
            remaining = candidates & table[r]
            if not remaining:
                logging.warning(f'{r} leaves us with nothing. Throwing out this line.')
            else:
              logging.info(f'{r} has {len(table[r])} matches. {len(candidates)} left. '
                           f'{", ".join(candidates) if len(candidates) < 10 else ""}')
              candidates = remaining

    for w in candidates:
        print(w)

if __name__ == '__main__':
    main()
