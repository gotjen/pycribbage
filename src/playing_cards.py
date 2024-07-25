
from random import sample, shuffle
from dataclasses import dataclass
from typing import Set, List, Union

# Card type def
SUIT = int
Suit = range(4)
suit_char = u'\u2660\u2661\u2662\u2663' # '♠♥♢♣'

VALUE = int
Value = range(13)
value_char = [' A',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9','10',' J',' Q',' K']

# immutable, hashable Card dataclass
@dataclass(eq=True, frozen=True)
class Card:
    '''Dataclass for a playing card with Suit and Value indecies'''
    Suit:  SUIT
    Value: VALUE
    def __post_init__(self) -> None:
        '' 'Validate card suit and value'''
        mesg  = f'Card suit out of range [0-3]: {self.Suit}' if self.Suit not in Suit else ''
        mesg += f'\nCard value out of ranger [0-12]: {self.Value}' if self.Value not in Value else ''
        assert not mesg, ValueError(mesg)
    def __str__(self) -> str:
        '''Formats <Value><Suit> like 4♥'''
        return value_char[self.Value] + suit_char[self.Suit]
    def __repr__(self) -> str:
        return f'Card({self.__str__()})'

Hand = Set[Card,]

def full_deck(bar:Union[Hand, Card] = Hand) -> Hand:
    '''
    Generate a full deck of cards
    `deck = full_deck()` gives a deck of cards ordered by value, grouped by suit

    bar: optionally provide a set of cards to exclude from the deck. Useful for calculating outcome statistics
    `deck = full_deck( bar=hand )` gives all the remaining cards in the deck excluding cards in the set `hand`
    '''
    
    # force barred cards to be a set
    try:
        bar = set(bar)
    except TypeError:
        bar = {bar}

    deck = { Card(s,v) for s in Suit for v in Value if Card(s,v) not in bar }
    return deck

def sort_cards(cards:Set[Card,]) -> List[Card,]:
    return sorted(cards, key=lambda c: c.Value)

def random_cards(n=5, pool:Set[Card,]=full_deck()):
    return set( sample(list(pool),n) )

import unittest

class TestCards(unittest.TestCase):
    def setUp(self):
        self.deck = list(full_deck())

    def test_fulldeck(self):
        assert len(self.deck) == 52, f'There are 52 cards in a deck not {len(self.deck)}'
        

        excl = [self.deck[k] for k in range(0,52,3)]
        deck_part = full_deck(bar=excl)
        assert len(deck_part) == 52-len(excl), 'Should have excluded n cards'
        assert not any([e in deck_part for e in excl]), 'Excluded cards appear in deck sample'

        subsample = random_cards(n=10, pool=excl)
        assert all([c in excl for c in subsample]), 'All cards in sample must be from pool'

if __name__=='__main__':
    unittest.main()