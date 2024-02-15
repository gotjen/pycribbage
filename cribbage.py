#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:37:03 2024

@author: Henry Gotjen
"""
import time
import tqdm
import numpy as np
import pandas as pd
import plotly.express as px

from dataclasses import dataclass
from typing import Tuple, Set
from random import sample, shuffle
from itertools import combinations, chain

# Card type defs
SUIT = int
Suit = range(4)
suit_char = '♠♥♦♣'

VALUE = int
Value = range(13)
value_char = [' A',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9','10',' J',' Q',' K']

@dataclass(eq=True, frozen=True)
class Card:
    Suit:  SUIT
    Value: VALUE
    def __post_init__(self):
        mesg  = f'Card suit out of range [0-3]: {self.Suit}' if self.Suit not in Suit else ''
        mesg += f'\nCard value out of ranger [0-12]: {self.Value}' if self.Value not in Value else ''
        assert not mesg, ValueError(mesg)
    def __str__(self):
        return value_char[self.Value] + suit_char[self.Suit]
    def __repr__(self):
        return f'Card({self.__str__()})'

def full_deck(bar:set=set(), do_shuffle:bool=False) -> Set[Card]:
    '''
    Generate a full deck of cards
    `deck = full_deck()` gives a deck of cards ordered by value, grouped by suit

    bar: optionally provide a set of cards to exclude from the deck. Useful for calculating outcome statistics
    `deck = full_deck( bar=hand )` gives all the remaining cards in the deck excluding cards in the set `hand`

    do_shuffle: bool
    '''

    deck = { Card(s,v) for s in Suit for v in Value if (not Card(s,v) in bar) }
    if do_shuffle:shuffle(deck)
    return deck

def sortcards(cards:Tuple[Card,]):
    return sorted(cards, key=lambda c: c.Value)

def random_hand(n=5):
    return sortcards(sample(full_deck(),n))

# cribbage specific rules
CribValue = [1,2,3,4,5,6,7,8,9,10,10,10,10]

breakdown_counter = \
    {'zero': 0, # count   
     '15': 0, # count
     'pair': 0, # count
     'run4': 0, # count
     'run3': 0, # count
     'run5': 0, # count
     'flush4': 0, # count
     'flush5': 0, # count
     'nibs': 0, # count
    }
def breakdown_counter_empty ():
    return breakdown_counter.copy()

class CribCountException(Exception):
    pass

def isrun(sub):
    return  all([k==(c.Value-sub[0].Value) for k,c in enumerate(sub)])
def isnibs(hand):
    for c in hand[1:]:
        if c.Value == 10 and c.Suit == hand[0].Suit:
            return True
    return False
def is15(sub):
    return sum([CribValue[c.Value] for c in sub]) == 15
def ispair(sub):
    # if len(sub)==0:
    #     print(sub)
    #     return sub[0].Value == sub[1].Value
    # return False
    return len(sub)==2 and sub[0].Value==sub[1].Value
def isflush(hand):
    flush = 0
    if all([c.Suit == hand[1].Suit for c in hand[2:]]):
        flush = 4
        if hand[0].Suit == hand[1].Suit:
            flush += 1
    return flush

def chantscore(bd):
    chant = []
    scorecalc = 2 * bd['15'] + \
                2 * bd['pair'] + \
                3 * bd['run3'] + \
                4 * bd['run4'] + \
                5 * bd['run5'] + \
                4 * bd['flush4'] + \
                5 * bd['flush5'] + \
                1 * bd['nibs']
    running = 0

    #  zero
    if bd['zero']:
        chant.append('ZERO!')
        #assert scorecalc == running, CribCountException('Zero marked but points scored!')
        #return chant, scorecalc# save time

    if bd['15']:
        chant.append('Fifteen ' + ' '.join([str(k) for k in range(2,2*bd['15']+1,2)]))
        running += 2* bd['15']

    counted_pairs = False
    if bd['run3']>1: # double(s) run
        counted_pairs = True
        if bd['pair'] == 3:
            running += 15
            chant.append(f'Triple run for {running}')
        elif bd['pair'] == 2:
            running += 16
            chant.append(f'Double double run for {running}')
        elif bd['pair'] == 1:
            running += 8
            chant.append(f'Double run for {running}')
        else:
            raise CribCountException('There shouldn\'t be multiple runs with no pair')
    elif bd['run3']:
        running += 3
        chant.append(f'Run of 3 for {running}')
    elif bd['run4']>1:
        counted_pairs = True
        assert bd['pair'] == 1, CribCountException('There should be a pair with this double run of 4')
        running += 10
        chant.append(f'Double run of 4 for {running}')
    elif bd['run4']:
        running += 4
        chant.append(f'Run of 4 for {running}')
    elif bd['run5']:
        running += 5
        chant.append(f'Run of 5 for {running}')
    
    if bd['pair'] and not counted_pairs:
        running += 2 * bd['pair']
        chant.append(('Two' if bd['pair']==2 else '') + f'Pair for {running}')
    
    if bd['flush4']:
        running += 4
        chant.append(f'Flush of 4 for {running}')
    elif bd['flush5']:
        running += 5
        chant.append(f'Flush of 5 for {running}')
    
    if bd['nibs']:
        running += 1
        chant.append(f'His nibs for {running}')

    if len(chant)>1: chant[-1] = 'and ' + chant[-1]
    chant = '\n'.join(chant)

    assert scorecalc == running, CribCountException('Miscount')
    return chant,scorecalc

def cribscore(hand):
    breakdown = breakdown_counter_empty()
    score = 0

    # jack
    if isnibs(hand):
        breakdown['nibs'] += 1
        score +=1;
        
    # flush
    if flush:=isflush(hand):
        breakdown['flush'+str(flush)] += 1
        score += flush
    
    # Run thorugh all card combos
    combs = chain(*[combinations(hand,k) for k in range(len(hand),1,-1)])
    minrunlen = 3
    for sub in combs:
        sublen = len(sub)
        sub = sorted(sub, key=lambda c: c.Value)

        # 15's
        if is15(sub):
            breakdown['15'] += 1
            score += 2
        
        # run
        if sublen>=minrunlen and isrun(sub):
            breakdown['run'+str(sublen)] += 1
            score += sublen
            minrunlen = sublen
            
        # pair
        if len(sub)==2 and ispair(sub):
            breakdown['pair'] += 1
            score += 2
        
        #print(sub)

    if score==0:
        breakdown['zero'] += 1
             
    return score, breakdown

def splithand(hand,idisc):
    keep = [ card for k,card in enumerate(hand) if not k in idisc]
    disc = [ hand[k] for k in idisc ]

    return keep, disc

def rand_hand_test():
    hand = random_hand(6)
    #hand = [Card(1,8), Card(2,8), Card(2,7), Card(2,6), Card(2,5)]
    #hand = [Card(0,k) for k in range(8,13)]
    #hand = [Card(0,4), Card(0,5), Card(1,6), Card(0,6), Card(1,7)]

    print(hand)

    score, bd = cribscore(hand)
    print(bd)
    chant, _ = chantscore(bd)

    print(chant)

def score_all_hands():
    all_hands = list( combinations( all_cards, 5) )
    nhands = len(all_hands)

    SCORE = np.zeros(nhands)
    BDOWNS = [ breakdown_counter_empty() ] * nhands

    for idx, hand in enumerate(all_hands):
        SCORE[idx], BDOWNS[idx] = cribscore(hand)

    BDOWN = breakdown_counter_empty()
    for key in BDOWN.keys():
        BDOWN[key] = np.array( [ bd[key] for bd in BDOWNS ] )

    return SCORE, BDOWN

def score_all_hands_parallel():
    from concurrent.futures import ProcessPoolExecutor

    all_hands = combinations( all_cards, 5)

    with ProcessPoolExecutor(max_workers=5) as pool:
        SCORE, BDOWNS = zip(*pool.map(cribscore, all_hands) )

    SCORE = np.array(SCORE)
    BDOWN = breakdown_counter_empty()
    for key in BDOWN.keys():
        BDOWN[key] = np.array( [ bd[key] for bd in BDOWNS ] )

    return SCORE, BDOWN

def random_hand_stats():    # statistics
    nsamp = 10000
    HANDS = []
    BDOWN = breakdown_counter_empty()
    SCORE = np.zeros(nsamp)

    for k in range(nsamp):
        hand = random_hand()
        HANDS.append(hand)
        sc, bd = cribscore(hand)

        SCORE[k] = sc

        #print(hand,bd)

        for key in BDOWN.keys():
            BDOWN[key] += bd[key]

    print(BDOWN)

    # sampmean = SCORE.mean()
    # figtit = f'Distribution of random Cribbage hands, n={nsamp}, mean={sampmean}'
    # fig1 = px.histogram(SCORE, title=figtit)
    # fig1.update_xaxes(title='Score')
    # fig1.update_yaxes(title='Frequency')
    # fig1.show()

    data = {'Source': list(BDOWN.keys()), 'Frequency':list(BDOWN.values())}
    fig2 = px.bar(data, x='Source',y='Frequency', title='Point Source')
    fig2.show()

def brute_cut_analysis(deal:Tuple[Card,]):

    ndeal = len(deal)
    DECK = full_deck(bar=deal) # remaining draw options

    iDISC = [keep for keep in combinations(range(ndeal),2)]
    iKEEP = [tuple([k for k in range(ndeal) if k not in idisc]) for idisc in iDISC]
    SCORE = np.zeros((len(DECK),len(iDISC)))

    
    # print(deal)
    for j, ikeep in enumerate(iKEEP):
        for k,cut in enumerate(DECK):
            hand = [cut] + [deal[k] for k in ikeep]
            SCORE[k,j] = cribscore(hand)[0]
    
    return vars() #SCORE, iDISC, iKEEP, DECK

def optimal_discard(deal:Tuple[Card,]):
    res = brute_cut_analysis(deal)
    SCORE = res['SCORE']
    iDISC = res['iDISC']
    iKEEP = res['iKEEP']
    DECK = res['DECK']
    
    # stats
    # score axes: score[ k:cut_of_deck, j:kept_hand]
    maxscores  = np.max(SCORE, axis=0)
    meanscores = np.mean(SCORE, axis=0)
    medscores  = np.median(SCORE, axis=0)
    minscores  = np.min(SCORE, axis=0)
    fitness = (maxscores + meanscores + medscores + minscores)/4

    ibest = fitness.argmax()
    keep = [ deal[k] for k in iKEEP[ibest] ]
    disc = [ deal[k] for k in iDISC[ibest] ]
    expectedscore = fitness[ibest]

    return keep, disc

def rand_deal_explorer():

    deal = random_hand(6)
    print('Dealt: ', deal)

    res = brute_cut_analysis(deal)
    SCORE = res['SCORE']
    iDISC = res['iDISC']
    iKEEP = res['iKEEP']
    DECK  = res['DECK']
    
    # stats
    # score axes: score[ k:cut_of_deck, j:kept_hand]
    maxscores  = np.max(SCORE, axis=0)
    meanscores = np.mean(SCORE, axis=0)
    medscores  = np.median(SCORE, axis=0)
    minscores  = np.min(SCORE, axis=0)
    fitness = maxscores + meanscores + medscores + minscores

    keepslice = lambda j: [ deal[k] for k in iKEEP[j] ]
    discslice = lambda j: [ deal[k] for k in iDISC[j] ]

    stats = pd.DataFrame({'fitness':fitness, 'maximum':maxscores,
                          'mean':meanscores, 'median':medscores, 
                          'minimum':minscores,
                          'keep':[ keepslice(k) for k in range(len(iKEEP))],
                          'discard': [ discslice(k) for k in range(len(iDISC))]})
    stats = stats.sort_values('fitness',ascending=False, ignore_index=True)

    # Highest possible score
    ikeepmax = maxscores.argmax()
    ibestcut = SCORE[:,ikeepmax].argmax()
    iworstcut = SCORE[:,ikeepmax].argmin()

    print(f'___________\nHighest possible score')
    print(f'With hand: {keepslice(ikeepmax)}')
    print(f'Best cut: {DECK[ibestcut]}, Score: {maxscores[ikeepmax]}')
    print(f'Mean score: {meanscores[ikeepmax]}, Median score: {medscores[ikeepmax]}')
    print(f'Worst cut: {DECK[iworstcut]}, Score {minscores[ikeepmax]}')

    # Highest mean score
    ikeepmean = meanscores.argmax()
    ibestcut = SCORE[:,ikeepmean].argmax()
    iworstcut = SCORE[:,ikeepmean].argmin()

    print(f'___________\nHighest mean score')
    print(f'With hand: {keepslice(ikeepmean)}')
    print(f'Best cut: {DECK[ibestcut]}, Score: {maxscores[ikeepmean]}')
    print(f'Mean score: {meanscores[ikeepmean]}, Median score: {medscores[ikeepmean]}')
    print(f'Worst cut: {DECK[iworstcut]}, Score {minscores[ikeepmean]}')

    # Highest median score
    ikeepmed = medscores.argmax()
    ibestcut = SCORE[:,ikeepmed].argmax()
    iworstcut = SCORE[:,ikeepmed].argmin()

    print(f'___________\nHighest Median score')
    print(f'With hand: {keepslice(ikeepmed)}')
    print(f'Best cut: {DECK[ibestcut]}, Score: {maxscores[ikeepmed]}')
    print(f'Mean score: {meanscores[ikeepmed]}, Median score: {medscores[ikeepmed]}')
    print(f'Worst cut: {DECK[iworstcut]}, Score {minscores[ikeepmed]}')

    # Highest worst case score
    ikeepmin = minscores.argmax()
    ibestcut = SCORE[:,ikeepmin].argmax()
    iworstcut = SCORE[:,ikeepmin].argmin()

    print(f'___________\nHighest Worst case score')
    print(f'With hand: {keepslice(ikeepmin)}')
    print(f'Best cut: {DECK[ibestcut]}, Score: {maxscores[ikeepmin]}')
    print(f'Mean score: {meanscores[ikeepmin]}, Median score: {medscores[ikeepmin]}')
    print(f'Worst cut: {DECK[iworstcut]}, Score {minscores[ikeepmin]}')

    ## fittness parameter
    ikeepfit = fitness.argmax()
    ibestcut = SCORE[:,ikeepfit].argmax()
    iworstcut = SCORE[:,ikeepfit].argmin()

    print(f'___________\nBest "fittness" parameter')
    print(f'With hand: {keepslice(ikeepfit)}')
    print(f'Best cut: {DECK[ibestcut]}, Score: {maxscores[ikeepfit]}')
    print(f'Mean score: {meanscores[ikeepfit]}, Median score: {medscores[ikeepfit]}')
    print(f'Worst cut: {DECK[iworstcut]}, Score {minscores[ikeepfit]}')

    print(ikeepmax, ikeepmean, ikeepmed, ikeepmin, ikeepfit)

    print(stats)

class CribAgent:
    strategy:str

    def __init__(self, strategy=None):
        self.strategy = strategy # [human, random, max, mean, fit]

    def choose(self,deal):
        match self.strategy:
            case 'human':
                return self.choose_human(deal)
            case 'random':
                return self.choose_random(deal)
            case 'fit':
                return self.choose_optimal(deal)
            case _:
                raise Exception(f'No such strategy {self.strategy}')
    def choose_human(self,deal):
        for k,card in enumerate(deal):
            print(k, card)
        
        while True:
            try:
                C = input('Choose discard. Separate index by space: ')
                C = np.asarray( C.split(' '), dtype=int)
                return splithand(deal, C)
            except Exception as err:
                if c == 'q':
                    raise err
                print('invalid')
    def choose_random(self,deal):
        return deal[:4], deal[4:6]
    
    def choose_optimal(self,deal):
        return optimal_discard(deal)

def cribmatch(p1_strat, p2_strat):

    playtoscore = 10000
    dealer = False
    agents = {True: CribAgent(p1_strat), False: CribAgent(p2_strat)}
    scoreboard = {True: 0, False: 0}

    nloop = 0; looplimit = playtoscore/4
    while nloop<looplimit:
        nloop += 1
        player = not dealer # oppostie player

        # deal the cards
        deck = full_deck(do_shuffle=True)
        deal = {True: deck[0:6], False: deck[6:12]}
        cut = deck[-1]

        # player agents choose keeps
        keep = {True: [], False: []}
        keep[dealer], disc1 = agents[dealer].choose(deal[dealer])
        keep[player], disc2 = agents[player].choose(deal[player])
        crib = disc1 + disc2

        # score player
        score, _ = cribscore( [cut] + keep[player] ); scoreboard[player] += score
        if scoreboard[player] >= playtoscore:
            winner = ('Player1' if player else 'Player2') + ':' + agents[player].strategy
            break

        # score dealer
        score, _ = cribscore( [cut] + keep[dealer] ); scoreboard[dealer] += score
        score, _ = cribscore( [cut] + crib ); scoreboard[dealer] += score
        if scoreboard[dealer] >= playtoscore:
            winner = ('Player1' if dealer else 'Player2') + ':' + agents[dealer].strategy
            break

        dealer = not dealer

    if nloop==looplimit:
        raise Exception('Game exited at loop limit')

    print('Winner! ', winner)
    print(f'Player1:{agents[True].strategy}: {scoreboard[True]}')
    print(f'Player2:{agents[False].strategy}: {scoreboard[False]}')


if __name__ == '__main__':

    # cribmatch('fit','fit')
    N = 5000

    agent = CribAgent()
    deck = full_deck()

    score_rand = np.zeros(N)
    bdown_rand = [ breakdown_counter_empty() ] * N

    score_fit = np.zeros(N)
    bdown_fit = [ breakdown_counter_empty() ] * N

    for k in range(N):

        samp = sample(list(deck), 7)
        cut = [ samp[0] ]
        deal = samp[1:]

        score_rand[k], bdown_rand[k] = cribscore(cut + agent.choose_random(deal)[0])
        score_fit[k] , bdown_fit[k]  = cribscore(cut + agent.choose_optimal(deal)[0])

    df =pd.DataFrame(dict(
        Strategy=np.concatenate((["Random"]*N, ["Optimal"]*N)), 
        Score  =np.concatenate((score_rand,score_fit))
    ))

    fig = px.histogram(df, x="Score", color="Strategy", barmode="overlay",
                        title=f'Optimal Cribbage Strategy vs Random Discard, N={N}')
    fig.show()

