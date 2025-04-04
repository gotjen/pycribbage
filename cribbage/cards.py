import re
from random import sample, shuffle
from dataclasses import dataclass
from typing import List, Tuple


# Card type def
SUIT = int
Suit = [SPADE, HEART, DIAMOND, CLUB] = range(4)
suit_char = u'\u2660\u2661\u2662\u2663'  # '♠♥♢♣'

VALUE = int
Value = [ACE,TWO,THREE,FOUR,FIVE,SIX,SEVEN,EIGHT,NINE,TEN,JACK,QUEEN,KING] = range(13)
value_char = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


# immutable, hashable Card dataclass
@dataclass(eq=True, frozen=True)
class Card:
    '''Dataclass for a playing card with Suit and Value indecies'''
    Suit:  SUIT
    Value: VALUE

    def __post_init__(self) -> None:
        '''Validate card suit and value'''
        mesg = f'Card suit out of range [0-3]: {self.Suit}' if self.Suit not in Suit else ''
        mesg += f'\nCard value out of ranger [0-12]: {self.Value}' if self.Value not in Value else ''
        assert not mesg, ValueError(mesg)

    @property
    def suit_char(self):
        return suit_char[self.Suit]

    @property
    def value_char(self):
        return value_char[self.Value]

    def __str__(self) -> str:
        '''Formats <Value><Suit> like [ 4 ♥ ]'''
        return '[' + ' '*(2-len(self.suit_char)) + self.suit_char + ' ' + self.value_char + ' ]'
    
    def __repr__(self) -> str:
        return self.__str__()


Hand = List[Card,]


def full_deck(bar:Hand=[], suits=Suit, values=Value, do_shuffle=False) -> Hand:
    '''
    Generate a full deck of cards
    `deck = full_deck()` gives a deck of cards ordered by value, grouped by suit

    bar: optionally provide a list of cards to exclude from the deck. Useful for calculating outcome statistics
    `deck = full_deck( bar=hand )` gives all the remaining cards in the deck excluding cards in the list `hand`

    suits,values: restrict the source of suits or values.
    `spades = full_deck(suits=[SPADE])` returns all the spades in the deck
    '''

    # force barred cards to be a list
    if isinstance(bar, Card):
        bar = [bar]

    deck = [Card(s, v) for s in suits for v in values if Card(s, v) not in bar]

    if do_shuffle:
        shuffle(deck)
    return deck


def sort_cards(cards:Hand) -> Hand:
    '''Sort cards by value'''
    return sorted(list(cards), key=lambda c: c.Value + 0.1*c.Suit)


def random_cards(n=5, pool:Hand=None) -> Hand:
    '''
    Generate unique hand of random cards
    n: number of random in cards.
    pool (optional): list of cards to choose from without replacement
    '''

    if pool is None:
        pool = full_deck()
    return sample(pool, n)

def str2hand(s:str) -> Hand:
    '''
    Parse cards from a string in the form '[ <suit><card ]'.
    Return a hand of cards
    '''  

    patt = r'\[\s*([♠♡♢♣])\s*(\d+|[AJQK])\s\]'
    matches = re.findall(patt,s)

    cards = []
    
    for S,V in matches:
        try:
            invalid = 'suit'
            s = cards.suit_char.index(S)
            invalid = 'valid'
            v = cards.value_char.index(V)
        except ValueError:
            print(f'Skipping invalid {invalid} in card [ {V},{S}]')

        this_card = Card(s,v)
        assert this_card not in cards, 'Illegal. Duplicate cards appear: {this_card}.'

        cards.append( this_card )

    return cards

def discard_from_hand(hand: Hand, idisc: List[int]) -> Tuple[Hand, Hand]:
    '''
    Split a hand into discard and keep piles according to the discard indices
    '''
    assert all( [i<len(hand) for i in idisc]), f"Discard indecies out of range {idisc}"
    keep = hand.copy()
    idisc_cleaned = sorted(list(set(idisc)), reverse=True)
    disc = [keep.pop(k) for k in idisc_cleaned]
    return keep, disc
