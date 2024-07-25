import unittest
import random

from src.playing_cards import Card, full_deck, sort_cards, random_cards
from src.cribbage_gameplay import breakdown_counter_empty, cribscore, isnibs, isflush, isflush_in_hand, isrun, isfifteen, ispair, chantscore

rand_suit = lambda: random.randint(0,3)

def gen_fifteens():
    k = 15
    maxlen = 5
    maxval = 10

class TestCribbage(unittest.TestCase):
    def setUp(self):
        pass

    def test_pairs(self):
        N = 100
        rand_val = [ random.randint(0,12) for _ in range(N) ]
        pairs = [[Card(Suit= rand_suit(),Value=v), Card(Suit= rand_suit(),Value=v)] for v in rand_val ]

        not_pairs = [[Card(Suit=rand_suit(), Value=v), Card(Suit=rand_suit(),Value=(v + random.randint(1,3))%4)] for v in rand_val ]

        assert all( map(ispair, pairs) ), 'Pair check funtion false negative'
        assert not any( map(ispair, not_pairs) ), 'Pair check function false positive'

    def test_runs(self):
        N = 100
        runlen = [ random.randint(3,5) for _ in range(N) ]
        runstart = [ random.randint(0,12-L) for L in runlen ]

        runs = [ [Card(Suit=rand_suit(), Value=S+k) for k in range(L)] for L,S in zip(runlen,runstart)]

        not_runs = []

        assert all( map(isrun,runs) ), 'Run check false negative'

    def test_fifteens(self):
        N = 100

        fifteens = [[Card(2,4),Card(1,11)]]

        assert all( map(isfifteen,fifteens) ), 'Fifteen check false negative'


if __name__=='__main__':
    unittest.main()
