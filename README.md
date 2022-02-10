I thought I'd try brute forcing Wordle, exhaustively searching the
entire game space and coming up with a provably optimal set of
plays for all situations.

# Design Choices

- We get to know the official word lists.

- We play each game independently. There's no history guiding the player or the host.

- **The player only guesses words that could be the answer.**

  This has two implications, each of which limit the player to
  choices that are probably worse than they need to be!

  1. The player is limited to the 2315 words that Wordle might pick, not
     the 12972 Wordle allows as legal guesses.

  2. The player only guesses words consistent with all revealed
     information so far, which is even more restrictive than "hard
     mode". (Wordle's hard mode requires you to use all confirmed letters,
     but you can use them in positions you know are wrong).

  These contribute to simple code structure and faster runtime, but
  they give up the goal of solving proper unrestricted Wordle.


- We optimize for "shortest games on average", ignoring the 6-guess limit entirely.

  I thought of a few other reasonable measures of success (include a
  penalty for losing, maximize number of games won in 4 moves, etc),
  but this one is simple and allows for better cache performance than
  depth-dependent metrics.



# Future Work

 Not that I plan to do any more, but just to point out what the obvious next steps would be.


## Faster

- ~1000x - deploy it on a cluster. The different starting words are essentially independent, so it would be relatively easy to distribute via mapreduce, AWS Lambda, or whatever.

- 10x - rewrite in C++

- ?? - better cache tuning & ordering. Choosing what size subtrees to cache, finding a more efficient way to store & load the cache to share across machines... The score cache is too big to use in its entirety for the entire 2315 word run, duplicated across multiple cores. Smaller subtrees are presumably repeated more, but larger ones would take longer to recompute. I did some rough analysis of this, but stopped short of applying it.

- ?? - prune the search


## More Comprehensive

- Let the player use a different word list.
- Play on easy mode; don't reduce the player's options based on new information. This will probably require pruning to be tractable. Order branches by estimated value, probability weighted, etc., only pursue the promising ones.



# Running it


The simplest way to run this is to just give it a word list and let it find the best soultion:

`wordle.py words-target`

But that would just output the score for the single best answer.

This is approximately what I ran to generate [the scores I got](https://github.com/sethoscope/wordle-brute/blob/main/output/scores):

`xargs -n 1 -P 32 pypy3 wordle.py --procs 1 words-target < words-target`

It took about 5.5hrs on a 32 core machine.


# Also

## Apex Predator Wordle!

Guess the word only using a bunch of grids of colored squares people
share! In my very limited testing, it looked like it took around 10-20
shared games to narrow it down to 1 word. [My solution](https://github.com/sethoscope/wordle-brute/blob/main/apexpredator.py) is not robust to
noisy input though.

Ben Hamner did [something similar](https://www.kaggle.com/benhamner/wordle-1-6) and has a more robust solution.

