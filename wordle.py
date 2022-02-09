#!/usr/bin/env python3
#
# Explore possible Wordle games and find the optimal guesses.
#
# 2022-01-18  Seth Golub <entropy@gmail.com>


import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
import pickle
from collections import defaultdict, ChainMap, UserList
from tempfile import NamedTemporaryFile
import os
import multiprocessing

# The way we organize this is as a two-player game. On the human's turn,
# she guesses a word. Then the computer/host plays by choosing a word
# that's consistent with all the feedback so far, and giving feedback
# about how the player's guess matches it. If it's a perfect match, the
# player wins.
#
# Depending on how the host chooses its word, this can be actual
# Wordle or it can be a different game (e.g. Absurdle). To mimic
# Wordle, the host choose a word at random with uniform probability,
# without considering the player's guess.
#
# The reason we do things this way, rather than choosing a secret word
# at the outset, is that it's simple to model and could be deployed to
# play a real game against a host whose word we truly don't know.
# If we wanted to add constraints like "use the word of the day", that
# could just as easily be applied by the host at each turn.
#
# There's one big divergence from actual Wordle though. I only have
# the player guess words that might potentially be the answer. This
# manifests in two ways:
#   1. I use the same word list for both Player and Host. This is
#      mostly about convenience and wouldn't be hard to change.
#   2. The player only guesses words that are consistent with all the
#      feedback. This is similar to playing Wordle in "hard mode" and
#      is probably not optimal! So that's a big caveat on whatever we
#      learn from this. But having ~13K options instead of ~2K would
#      probably impact running time significantly, so I've left it
#      like this for now. Further optimizations, parallelism, or
#      search pruning could easily make this feasible.


class Histogram(UserList):
    '''
    Like Counter, but indexes will just be small ints, and we fill gaps.
    We use this to see how many guesses all the subsequent games take.
    '''
    def __getitem__(self, i):
        if i >= len(self):
            self.extend([0] * (i - len(self) + 1))
        return UserList.__getitem__(self, i)

    def __setitem__(self, i, v):
        if i >= len(self):
            self.extend([0] * (i - len(self) + 1))
        UserList.__setitem__(self, i, v)

    def update(self, other):
        for i in range(len(other)):
            self[i] += other[i]

    def shift_right(self):
        self.data = [0] + self.data

    def to_chart(self, width=20):
        maxval = max(self.data)
        width = min(width, maxval)
        return '\n'.join(f'{i}: '
                         + ('*' * int(self[i] * width / maxval))
                         + (f'  ({self[i]})' if self[i] else '')
                         for i in range(1, len(self)))


class Evaluation():
    '''
    This is where we store anything we learned from evaluating a game state.
    '''
    def __init__(self, score=0, best_word=None, histogram=None):
        self.score = score
        self.best_word = best_word
        self.histogram = histogram or Histogram()

    def __lt__(self, other):
        return self.score < other.score


class PlayerScoreCache(ChainMap):
    '''
    We can cache the subtree score values and save ourselves a lot of
    time.  We assume we are using a metric that is independent of how
    we got here, so the state is defined only by which words remain as
    possible solutions. A metric that cared about whether we would solve
    the game within six guesses (for example) would need to include
    "guesses remaining" in the game state and the cache would not
    perform as well.

    The cache is not shared between processes.
   '''
    def __init__(self, *args):
        super().__init__({}, *args)

    def add(self, wordlist, evaluation):
        if evaluation.score > self.BIGGISH:
            return
            # We don't cache subtrees with losing games because we might
            # reach that same state by a shorter route and it would be
            # wrong to use the large score for that.  Ideally we'd build
            # the entire score cache using unbounded searches.
        self[wordlist] = evaluation

    def save_all(self, filename):
        logging.debug('Saving entire player score cache.')
        with open(filename, 'wb') as f:
            pickle.dump(dict(self), f)

    def save_new(self, filename):
        logging.debug('Saving player score cache updates.')
        with open(filename, 'wb') as f:
            pickle.dump(self.maps[0], f)

    def load(self, filenames):
        self.maps = [{}]
        for filename in filenames:
            try:
                with open(filename, 'rb') as f:
                    logging.debug('Loading player score cache.')
                    self.maps.append(pickle.load(f))
            except FileNotFoundError:
                logging.warning(f'Cache file {filename} not found.')


class Response():
    """
    Represents the host's move in the game: feedback about the letters
    in the player's guess.
    """

    ABSENT = 0
    PRESENT = 1
    CORRECT = 2

    DEBUGCHAR = {0: '.',
                 1: 'x',
                 2: 'O'}

    def __init__(self, tags):
        self.tags = tuple(tags)

    @classmethod
    def from_guess(cls, target, guess):
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
                        target_avail[j] = False
        return cls(result)

    def all_correct(self):
        return all(t == self.CORRECT for t in self.tags)

    def __eq__(self, other):
        return self.tags == other.tags

    def __hash__(self):
        return hash(self.tags)

    def __str__(self):
        return ''.join(self.DEBUGCHAR[t] for t in self.tags)


class WordList(frozenset):
    def filter(self, guess, response):
        '''Return a new WordList consistent with guess & response.'''
        # The simplest thing is to make a Response for each word using
        # the guess and see if it matches what we're given. It's not
        # very fast though, so we do other things to rule out some
        # words.
        # TODO: is these helping enough to bother?
        must = set(L for i, L in enumerate(guess)
                   if response.tags[i] != Response.ABSENT)
        mustnot = set(L for i, L in enumerate(guess)
                      if response.tags[i] == Response.ABSENT and L not in must)
        matches = [(i, L) for i, L in enumerate(guess)
                   if response.tags[i] == Response.CORRECT]
        return self.__class__(w for w in self.words
                              if (must <= set(w) and
                                  mustnot.isdisjoint(set(w)) and
                                  all(w[i] == L for (i, L) in matches) and
                                  Response.from_guess(w, guess) == response))


class Host():
    '''
    The opponent, who chooses a word the player tries to guess.
    '''
    def score_position(self, wordlist, player_guess, player, depth, max_depth):
        '''
        Recurse through all possible games from here and return
        the average score of those games.
        '''
        # First we figure out all possible responses, along with how
        # many words yield each one. Then we'll follow each one just
        # once and multiply.
        by_response = defaultdict(set)
        for w in wordlist:
            by_response[Response.from_guess(w, player_guess)].add(w)
        ev = Evaluation(0.0)
        for response, words in by_response.items():
            pev = player.score_position(WordList(words),
                                        response,
                                        self,
                                        depth, max_depth)
            if depth <= debug_host_depth:
                logging.debug(f'H{depth}  {". "*depth}'
                              f'{response}:{len(words)} : {ev.score:.5f}')
            ev.score += len(words) * pev.score / len(wordlist)
            ev.histogram.update(pev.histogram)

        if depth <= debug_host_depth:
            logging.debug(f'H{depth}  {". "*depth} score {ev.score:.5f}')
        return ev


# Rather than make the Host know about depth & max_depth, we could give
# it a callback and bake the depth in with a lambda closure. Then only
# the player code has to see it. But it's nice to have it for debug
# output.


class Player():
    '''
    The player, who makes a guess (and evaluates the utility of game states).
    '''

    BIGNUM = 1000000   # penalty for not winning

    def __init__(self):
        self.score_cache = PlayerScoreCache()
        self.score_cache.BIGGISH = 1000   # bigger must include a penalty

    # We do this because we can't use lambda with multiprocessing
    class _BoundHostCall():
        def __init__(self, player, host, wordlist, depth, max_depth):
            self.player = player
            self.host = host
            self.wordlist = wordlist
            self.depth = depth
            self.max_depth = max_depth

        def __call__(self, word):
            return (self.host.score_position(self.wordlist, word, self.player,
                                             self.depth, self.max_depth),
                    word)

    def score_position(self, wordlist, host_response, host, depth, max_depth,
                       guess=None, procs=1):
        '''
        Recurse through all possible games from here and return the
        score of the path we would choose to take.  A guess can be
        provided, in which case we only try that one.
        '''
        if host_response and host_response.all_correct():   # got it last time
            return Evaluation(0, '', Histogram((1,)))
        if depth == max_depth:
            return Evaluation(self.BIGNUM, '', Histogram())  # penalize losing
        try:
            return self.score_cache[wordlist]
        except KeyError:
            pass
        depth += 1
        guess_list = [guess] if guess else wordlist
        get_ev = self._BoundHostCall(self, host, wordlist, depth, max_depth)
        procs = min(procs, len(guess_list))
        if procs <= 1:
            ev, best_word = min(map(get_ev, guess_list))
        else:
            with multiprocessing.Pool(procs) as pool:
                ev, best_word = min(pool.map(get_ev, guess_list))
        if depth <= debug_player_depth:
            logging.debug(f'P{depth}  {". "*depth}'
                          f'best word: {best_word} ({ev.score:.5f})')
        ev.best_word = best_word
        ev.score += 1
        ev.histogram.shift_right()
        if not guess:  # If we only tried one word, we can't score this state
            self.score_cache.add(wordlist, ev)
        return ev

    def start(self, wordlist, host, max_depth, guess, procs):
        return self.score_position(wordlist, None, host, 0, max_depth,
                                   guess, procs)


def main():
    logging.basicConfig(format='%(relativeCreated)8d ms  // %(message)s')
    description = 'wordle solver'
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--maxdepth', type=int, default=9999)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--histogram', action='store_true')
    parser.add_argument('--histogram_width', type=int, default=72)
    parser.add_argument('--cache_in', metavar='FILENAME',
                        nargs='+',
                        help='input score cache file(s)')
    parser.add_argument('--cache_out', metavar='FILENAME',
                        help='output score cache entries')
    parser.add_argument('--cache_out_updates', metavar='FILENAME',
                        help='output only new score cache entries')
    parser.add_argument('--procs', type=int,
                        default=multiprocessing.cpu_count(),
                        help='number of parallel processes')
    parser.add_argument('--debug_player_depth', type=int, default=6)
    parser.add_argument('--debug_host_depth', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('wordfile', type=FileType('r'))
    parser.add_argument('startword', nargs="?")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global debug_host_depth, debug_player_depth
    debug_player_depth = args.debug_player_depth
    debug_host_depth = args.debug_host_depth

    wordlist = WordList(line.strip() for line in args.wordfile.readlines())

    player = Player()
    if args.cache_in:
        player.score_cache.load(args.cache_in)
    ev = player.start(wordlist, Host(), args.maxdepth, args.startword,
                      args.procs)
    print(f'{ev.score:.5f} {args.startword or ev.best_word}')
    if args.histogram:
        print(ev.histogram.to_chart(args.histogram_width))
    if args.cache_out:
        player.score_cache.save_all(args.cache_out)
    if args.cache_out_updates:
        player.score_cache.save_new(args.cache_out_updates)


if __name__ == '__main__':
    main()
