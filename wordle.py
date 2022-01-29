#!/usr/bin/env python3
#
# This code will simulate and solve Wordle games.  In fact, it's
# playing a slightly different game, in which the computer opponent
# chooses a new word at each step, but with the word choice algorithm,
# it's indistinguishable from Wordle (and it's easier to model).
#
# 2022-01-18  Seth Golub <entropy@gmail.com>


import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType
from collections import defaultdict

WORDLEN = 5


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


class PlayerScoreCache(dict):
    def __init__(self):
        self.hits = 0
        self.tests = 0
        super().__init__(self)
    
    def get(self, wordlist):
        v = super().get(wordlist, None)
        if v:
            self.hits += 1
        self.tests += 1
        if self.tests >= 100000:
            logging.debug(f'recent cache hits: {int(100 * self.hits / self.tests)}%')
            self.tests = 0
            self.hits = 0
        return v

    def add(self, wordlist, score):
        if score > self.BIGGISH:
            return
            # We don't cache subtrees with losing games because we might
            # reach that same state by a shorter route and it would be
            # wrong to use the large score for that.  Ideally we'd build
            # the entire score cache using unbounded searches.
        self[wordlist] = score


class Response():
    """
    Represents the host's move in the game: feedback about the letters
    in the player's guess.
    """

    ABSENT = 0
    PRESENT = 1
    CORRECT = 2

    DEBUGCHAR = {0 : '.',
                 1 : 'x',
                 2 : 'O'}

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
                        target_avail[j] = False;
        return cls(result)
    
    def all_correct(self):
        return all(t == self.CORRECT for t in self.tags)

    def __eq__(self, other):
        return self.tags == other.tags

    def __hash__(self):
        return hash(self.tags)

    def __str__(self):
        return ''.join(self.DEBUGCHAR[t] for t in self.tags)


class WordList():
    def __init__(self, words):
        self.words = frozenset(words)

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __hash__(self):
        return hash(self.words)

    def __eq__(self, other):
        return self.words == other.words

    def filter(self, guess, response):
        '''Return a new WordList consistent with guess & response.'''
        # The simplest thing is to make a Response for each word using
        # the guess and see if it matches what we're given. It's not
        # very fast though, so we do other things to rule out some
        # words.
        must = set(L for i,L in enumerate(guess) if response.tags[i] != Response.ABSENT)
        mustnot = set(L for i,L in enumerate(guess) if response.tags[i] == Response.ABSENT and L not in must)
        matches = [(i,L) for i,L in enumerate(guess) if response.tags[i] == Response.CORRECT]
        return self.__class__(w for w in self.words
                              if (must <= set(w) and
                                  mustnot.isdisjoint(set(w)) and
                                  all(w[i] == L for (i,L) in matches) and
                                  Response.from_guess(w, guess) == response))


class Host():
    '''
    The opponent
    '''
    def score_position(self, wordlist, player_guess, player, depth, max_depth):
        '''
        Recurse through all possible games from here and return
        probability-weighted average score of those games.
        (For standard Wordle, all words are equally likely.)
        '''
        # First we figure out all possible responses, along with how
        # many words cause each one. Then we'll follow each one just
        # once and multiply.

        # optimization: at maxdepth, we can just return 1/N
#         if depth == max_depth:
#             return 1 / len(wordlist)
        
        # Rather than explore the game tree for each word we could
        # choose, we group words that yield the same response.
        by_response = defaultdict(set)
        #n, N = 0, len(wordlist)
        for w in wordlist:
            #n += 1
            #logging.debug(f'H{depth} {int(100*n/N)}%  {". "*depth}  response for {w}')
            by_response[Response.from_guess(w, player_guess)].add(w)
        total = 0.0
        n, N = 0, len(by_response)
        for response,words in by_response.items():
            n += 1
            # TODO: parallelize, even if just at depth 1
            score = player.score_position(WordList(words),
                                          response,
                                          self,
                                          depth, max_depth)
            if depth <= debug_host_depth:
                logging.debug(f'H{depth} {int(100*n/N)}%  {". "*depth} {response}:{len(words)} : {score:.5f}')
            total += len(words) * score
        score = total / len(wordlist)
        
        if depth <= debug_host_depth:
            logging.debug(f'H{depth}  {". "*depth} score {score:.5f}')
        return score


# Rather than make the Host know about depth & max_depth, we could give
# it a callback and bake the depth in with a lambda closure. Then only
# the player code has to see it. But it's nice to have it for debug
# output.


class Player():
    '''
    Players choose a word that maximizes something. Some examples of
    possible metrics:
     - maximize likelihood of winning in 6 turns
     - minimize mean number of turns
     - minimize number of turns in best 90% of games
     '''

    BIGNUM = 1000000   # penalty for not winning

    def __init__(self):
        self.score_cache = PlayerScoreCache()
        self.score_cache.BIGGISH =   1000   # anything bigger than this includes a penalty

    def score_position(self, wordlist, host_response, host, depth, max_depth, guess=None):
        '''
        Recurse through all possible games from here and return
        probability-weighted average score of those games.
        If guess is provided, use that instead of trying all possibilities.
        '''
        if host_response and host_response.all_correct():   # we got it last time
            return 0
        if len(wordlist) == 1:   # let's not go through all the steps
            return 1
        if depth == max_depth:
            return self.BIGNUM       # winning is important
        score = self.score_cache.get(wordlist)
        if score:
            return score
        depth += 1
        guess_list = [guess] if guess else wordlist
        best_word, best_score = None, None
        n, N = 0, len(guess_list)
        for word in guess_list:
            n += 1
            score = host.score_position(wordlist, word, self, depth, max_depth)
            if depth <= debug_player_depth:
                logging.debug(f'P{depth} {int(100*n/N)}%  {". "*depth}  {word} : {score:.5f}')
            if (best_word is None) or score < best_score:
                best_word = word
                best_score = score
        if depth <= debug_player_depth:
            logging.debug(f'P{depth}  {". "*depth}best word: {best_word} ({best_score:.5f})')
        score = best_score + 1
        self.score_cache.add(wordlist, score)
        return score

    def start(self, wordlist, host, max_depth, guess):
        return self.score_position(wordlist, None, host, 0, max_depth, guess)


# Score is likelihood of winning within the depth searched.
# There are other options:
#   highest likelihood that P% of games will end in N more turns
#   smallest weighted mean number of remaining moves to win
#   smallest median number of remaining moves to win




# TODO: allow caller to specify a word for the player to guess first.
# TODO: allow caller to provide a game so far, guesses and responses.

def main():
    description = ''
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--maxdepth', type=int, default=6)
    parser.add_argument('-v', '--verbose', action='store_true')
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
    score = player.start(wordlist, Host(), args.maxdepth, args.startword)
    print(f'{score:.5f} {args.startword or ""}')

if __name__ == '__main__':
    main()
