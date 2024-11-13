import unittest

from cribbage import *

class TestCards(unittest.TestCase):
    def setUp(self):
        pass

    def test_fulldeck(self):
        deck = full_deck()

        assert len(deck) == 52, f'There are 52 cards in a deck not {len(deck)}'
        
        excl = [deck[k] for k in range(0, 52, 3)]
        deck_part = full_deck(bar=excl)
        assert len(deck_part) == 52-len(excl), 'Should have excluded n cards'
        assert not any([e in deck_part for e in excl]), 'Excluded cards appear in deck sample'

        subsample = random_cards(n=10, pool=excl)
        assert all([c in excl for c in subsample]), 'All cards in sample must be from pool'


if __name__ == '__main__':
    unittest.main()