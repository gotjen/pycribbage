# pyCribbage

Cribbage is a phenomenal pub game - a thrilling blend of strategy and chance. Perfect past time while listening to music or catching up with a friend. But often I wonder what the best strategy is, how much better my play can be than making random choices and how rare are certain hands.

pyCribbage is a full implementation of the game and rules of cribbage, according to ools for generating and evaluating cribbage hands and statistics

## Components

- Playing card types and baseic methods found in `src/playing_cards`
- Cribbage gameplay implementation
  - Hand scoring and text output of hand counting the way real players speak
  - Scoring of the play, such as point for runs, pairs, fifteens and 31 in the play.
  - Class to implement the state and flow of a match between two opponents.
- Computer agent which makes optimal decisions
  - Optimal discard strategy based on crib and cut statistics
  - Optimal play strategy to maximize points and minimize opponent's score potential
 
- Tools for comaparing different strategies and their effectiveness over a large sameple size of games. 

Find  usage examples in `Examples` folder

Extendable and tweakable. Get in touch if you have ideas to share!
