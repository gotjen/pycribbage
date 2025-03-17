import unittest, pytest
from cribbage.cards import *

class TestCards(unittest.TestCase):
    def setUp(self):
        self.valid_cards: Hand = []
        print(self.valid_cards)
        self.deck = []

    def test_card(self):
        # Generate random valid cards
        # print(SPADE,DIAMOND,HEART,CLUB)
        # print(ACE,TWO,THREE,FOUR,FIVE,SIX,SEVEN,EIGHT,NINE,TEN,JACK,QUEEN,KING)

        self.valid_cards.append(Card(SPADE,ACE))
        self.valid_cards.append(Card(DIAMOND,10))
        self.valid_cards.append(Card(HEART,5))
        self.valid_cards.append(Card(CLUB,12))

        self.assertEqual(self.valid_cards[0].value_char, 'A')
        self.assertEqual(self.valid_cards[1].suit_char, u'\u2662')

        # invalid_cards
        with pytest.raises(Exception):
            c = Card(-1,  4) # SUIT out of range
        with pytest.raises(Exception):    
            c = Card( 4,  9) # SUIT out of range
        with pytest.raises(Exception):    
            c = Card( 2, -1) # VALUE out of range
        with pytest.raises(Exception):    
            c = Card( 3, 15) # VALUE out of range

    def test_fulldeck(self):
        # fulldeck
        self.deck = full_deck()
        self.assertEqual( len(self.deck), 52 )

        # fulldeck with exclusion
        deck_part = full_deck(bar=self.valid_cards)
        self.assertEqual(len(deck_part), 52-len(self.valid_cards))
        for c in self.valid_cards:
            self.assertNotIn(c,self.valid_cards)
    
    def test_randomcards(self):
        # Check no duplicates
        rand_hand = random_cards(5)
        self.assertEqual(len(rand_hand), len(set(rand_hand)))

    def test_sortcards(self):
        pass


if __name__ == '__main__':
    unittest.main()