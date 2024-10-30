"""
Created on Fri Jan  5 18:37:03 2024

@author: Henry Gotjen
"""

import time
# import tqdm
import numpy as np
import pandas as pd
import plotly.express as px

from typing import Tuple, Set, List
from random import sample, shuffle
from itertools import combinations
from src.playing_cards import Card, full_deck, sort_cards, random_cards
from src.cribbage_gameplay import breakdown_counter_empty, cribscore

def split_cards(hand:Set[Card,],idisc:List[int]) -> Tuple[Set[Card,],Set[Card,]]:
    keep = { card for k,card in enumerate(hand) if not k in idisc}
    disc = { hand[k] for k in idisc }

    return keep, disc

def rand_hand_test():
    hand = random_cards(6)
    #hand = [Card(1,8), Card(2,8), Card(2,7), Card(2,6), Card(2,5)]
    #hand = [Card(0,k) for k in range(8,13)]
    #hand = [Card(0,4), Card(0,5), Card(1,6), Card(0,6), Card(1,7)]

    print(hand)

    score, bd = cribscore(hand)
    chant = chantscore(bd)

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

def random_card_stats():    # statistics
    nsamp = 10000
    HANDS = []
    BDOWN = breakdown_counter_empty()
    SCORE = np.zeros(nsamp)

    for k in range(nsamp):
        hand = random_cards()
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
    DECK  = res['DECK']
    
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

    deal = random_cards(6)
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
                return split_cards(deal, C)
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
        player = not dealer # opposite player

        # deal the cards
        deck = full_deck(do_shuffle=True)
        deal = {True: deck[0:6], False: deck[6:12]}
        cut = deck[-1]

        # player agents choose keeps
        keep = {True: [], False: []}
        keep[dealer], discard1 = agents[dealer].choose(deal[dealer])
        keep[player], discard2 = agents[player].choose(deal[player])
        crib = discard1 + discard2

        # score player
        score, _ = cribscore( [cut] + keep[player] )
        scoreboard[player] += score
        if scoreboard[player] >= playtoscore:
            winner = ('Player1' if player else 'Player2') + ':' + agents[player].strategy
            break

        # score dealer
        score, _ = cribscore( [cut] + keep[dealer] )
        scoreboard[dealer] += score
        score, _ = cribscore( [cut] + crib )
        scoreboard[dealer] += score
        if scoreboard[dealer] >= playtoscore:
            winner = ('Player1' if dealer else 'Player2') + ':' + agents[dealer].strategy
            break

        dealer = not dealer

    if nloop==looplimit:
        raise Exception('Game exited at loop limit')

    print('Winner! ', winner)
    print(f'Player1:{agents[True].strategy}: {scoreboard[True]}')
    print(f'Player2:{agents[False].strategy}: {scoreboard[False]}')

def crib_score_stats():
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


if __name__ == '__main__':
    print('dingus')
