"""
Created on Fri Jan  5 18:37:03 2024

@author: Henry Gotjen
"""

# import time
# import tqdm
import numpy as np
import pandas as pd
import plotly.express as px

from typing import Tuple, List
from collections import Counter
from random import sample
from itertools import combinations

from cribbage.cards import full_deck, random_cards, discard_from_hand
from cribbage.game import cribscore,score_value, sum_cards, card_value, score_play

def rand_hand_test():
    hand = random_cards(6)
    # hand = [Card(1,8), Card(2,8), Card(2,7), Card(2,6), Card(2,5)]
    # hand = [Card(0,k) for k in range(8,13)]
    # hand = [Card(0,4), Card(0,5), Card(1,6), Card(0,6), Card(1,7)]

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
        for key in BDOWN.keys():
            BDOWN[key] += bd[key]

    sampmean = SCORE.mean()
    figtit = f'Distribution of random Cribbage hands, n={nsamp}, mean={sampmean}'
    fig1 = px.histogram(SCORE, title=figtit, barmode='relative')
    fig1.update_xaxes(title='Score')
    fig1.update_yaxes(title='RelativeFrequency')
    fig1.show()

    data = {'Source': list(BDOWN.keys()), 'Frequency':list(BDOWN.values())}
    fig2 = px.bar(data, x='Source',y='Frequency', title='Point Source')
    fig2.show()

def optimal_play(hand, inplay):
    '''
    Choose the best play to score points during phase 2 play
    '''
    # say 'go' if no cards can be played
    max_play_value = score_value['max_tally'] - sum_cards(inplay)
    iplayable = list(filter(lambda k: card_value[hand[k].Value] <= max_play_value, range(len(hand))))

    if not iplayable:
        return 'GO'

    if len(iplayable) == 1:
        return iplayable[0]

    # Try playing all cards to see if there are points
    play_score = list(map(score_play, [inplay + [hand[k]] for k in iplayable]))
    if any(play_score):
        return iplayable[play_score.index(max(play_score))]  # play the highest scoring card

    # TODO add more strategy
    for k in iplayable[::-1]:
        is_5 = hand[k].value_char == '5'
        makes_21 = sum_cards(inplay + [hand[k]]) == 21
        if not is_5 and not makes_21:
            return k

    return iplayable[-1]

def brute_cut_analysis(deal, is_dealer:bool=None):
    '''
    Analyse cribbage discard options by testing the statistics all options against all possible cuts.
    is_dealer: if set to True or False,
    '''
    ndeal = len(deal)

    CUTS = full_deck(bar=deal) # remaining draw options

    iDISC = list( combinations(range(ndeal), 2) )
    SCORE = np.zeros((len(CUTS),len(iDISC)))

    dof = 3
        
    for j, idisc in enumerate(iDISC):
        keep, disc = discard_from_hand( deal, idisc)

        discard_offset = 0
        if is_dealer is not None:
            # Account for value of discard in crib
            
            # straight-up points
            if sum_cards(disc) == 15:
                discard_offset += score_value['fifteen']

            # Neighboring cards can contribute to a run or pair
            match np.abs(disc[1].Value - disc[0].Value):
                # probability of giving a run to the crib
                case 0:
                    discard_offset += score_value['pair']  # definite pair
                case 1:
                    discard_offset += score_value['run'] * 3 * ( 1 - ((52-8)/52)**dof )  
                case 2:
                    discard_offset += score_value['run'] * 3 * ( 1 - ((52-4)/52)**dof )
                case _:
                    pass
            
            # Giving some easy scoring cards is bad
            for c in disc:
                if c.value_char == 'J':
                    # probability that jack give nibs in the crib
                    discard_offset += score_value['nibs'] * 0.25
                if c.value_char == '5':
                    # probabily that a five give fifteen in the crib
                    discard_offset += score_value['fifteen'] * ( 1- ((52-16)/52)**dof ) 

            if not is_dealer:
                discard_offset = -discard_offset

        for k, cut in enumerate(CUTS):
            hand = [cut] + keep
            SCORE[k,j] = cribscore(hand)[0] + discard_offset
    
    return SCORE, iDISC, CUTS

def optimal_discard(deal, is_dealer:bool=None):
    '''Calculate statistics from brute_cut_analysis, choose best discard'''
    SCORE, iDISC, CUTS = brute_cut_analysis(deal,is_dealer=is_dealer)
    
    # stats
    # score axes: score[ k:cut_of_deck, j:kept_hand]
    maxscores  = np.max(SCORE, axis=0)
    meanscores = np.mean(SCORE, axis=0)
    medscores  = np.median(SCORE, axis=0)
    minscores  = np.min(SCORE, axis=0)
    fitness = (maxscores + meanscores + medscores + minscores)/4

    ibest = fitness.argmax()
    keep, disc = discard_from_hand(deal, iDISC[ibest])
    expectedscore = fitness[ibest]

    # TODO add crib and play effects. Use Scoreplay to test.

    return iDISC[ibest]

def random_deal_explorer(deal = None, is_dealer:bool=None):

    if not deal:
        deal = random_cards(6)

    print('Dealt: ', deal)

    SCORE, iDISC, CUTS = brute_cut_analysis(deal, is_dealer=is_dealer)
    
    # stats
    # score axes: score[ k:cut_of_deck, j:kept_hand]
    maxscores  = np.max(SCORE, axis=0)
    meanscores = np.mean(SCORE, axis=0)
    medscores  = np.median(SCORE, axis=0)
    minscores  = np.min(SCORE, axis=0)
    fitness = (maxscores + meanscores + medscores + minscores)/4

    stats = pd.DataFrame({'fitness':fitness, 'maximum':maxscores,
                          'mean':meanscores, 'median':medscores, 
                          'minimum':minscores, 'i_discard': iDISC})

    stats = stats.sort_values('fitness',ascending=False, ignore_index=True)

    # # Highest possible score
    # ikeepmax = maxscores.argmax()
    # ibest = SCORE[:,ikeepmax].argmax()
    # iworst = SCORE[:,ikeepmax].argmin()
    # this_hand, this_disc = discard_from_hand(deal, iDISC[ikeepmax])

    # print(f'___________\nHighest possible score')
    # print(f'With hand: {this_hand}, and discard {this_disc}')
    # print(f'Best cut: {CUTS[ibest]}, Score: {maxscores[ikeepmax]}')
    # print(f'Mean score: {meanscores[ikeepmax]}, Median score: {medscores[ikeepmax]}')
    # print(f'Worst cut: {CUTS[iworst]}, Score {minscores[ikeepmax]}')

    # # Highest mean score
    # ikeepmean = meanscores.argmax()
    # ibest = SCORE[:,ikeepmean].argmax()
    # iworst = SCORE[:,ikeepmean].argmin()
    # this_hand, this_disc = discard_from_hand(deal, iDISC[ikeepmean])

    # print(f'___________\nHighest mean score')
    # print(f'With hand: {this_hand}, and discard {this_disc}')
    # print(f'Best cut: {CUTS[ibest]}, Score: {maxscores[ikeepmean]}')
    # print(f'Mean score: {meanscores[ikeepmean]}, Median score: {medscores[ikeepmean]}')
    # print(f'Worst cut: {CUTS[iworst]}, Score {minscores[ikeepmean]}')

    # # Highest median score
    # ikeepmed = medscores.argmax()
    # ibest = SCORE[:,ikeepmed].argmax()
    # iworst = SCORE[:,ikeepmed].argmin()
    # this_hand, this_disc = discard_from_hand(deal, iDISC[ikeepmed])

    # print(f'___________\nHighest Median score')
    # print(f'With hand: {this_hand}, and discard {this_disc}')
    # print(f'Best cut: {CUTS[ibest]}, Score: {maxscores[ikeepmed]}')
    # print(f'Mean score: {meanscores[ikeepmed]}, Median score: {medscores[ikeepmed]}')
    # print(f'Worst cut: {CUTS[iworst]}, Score {minscores[ikeepmed]}')

    # # Highest worst case score
    # ikeepmin = minscores.argmax()
    # ibest = SCORE[:,ikeepmin].argmax()
    # iworst = SCORE[:,ikeepmin].argmin()
    # this_hand, this_disc = discard_from_hand(deal, iDISC[ikeepmin])

    # print(f'___________\nHighest Worst case score')
    # print(f'With hand: {this_hand}, and discard {this_disc}')
    # print(f'Best cut: {CUTS[ibest]}, Score: {maxscores[ikeepmin]}')
    # print(f'Mean score: {meanscores[ikeepmin]}, Median score: {medscores[ikeepmin]}')
    # print(f'Worst cut: {CUTS[iworst]}, Score {minscores[ikeepmin]}')

    ## fittness parameter
    ikeepfit = fitness.argmax()
    ibest = SCORE[:,ikeepfit].argmax()
    iworst = SCORE[:,ikeepfit].argmin()
    this_hand, this_disc = discard_from_hand(deal, iDISC[ikeepfit])

    print(f'___________\nBest "fittness" parameter')
    print(f'With hand: {this_hand}, and discard {this_disc}')
    print(f'Best cut: {CUTS[ibest]}, Score: {maxscores[ikeepfit]}')
    print(f'Mean score: {meanscores[ikeepfit]}, Median score: {medscores[ikeepfit]}')
    print(f'Worst cut: {CUTS[iworst]}, Score {minscores[ikeepfit]}')

    return deal

class CribAgent:
    strategy:str
    name:str

    def __init__(self, strategy, name=None):
        self.strategy = strategy # [human, random, max, mean, fit]
        self.name = name

    def discard(self,deal, is_dealer:bool=None):
        match self.strategy:
            case 'human':
                for k,card in enumerate(deal):
                    print(k, card)
                
                while True:
                    try:
                        C = list(map(int, input('Choose discard: ').strip().replace(' ','')))
                        assert len(C) == 2, "Must discard two cards"
                        return C
                    except Exception as err:
                        if C == 'q':
                            raise KeyboardInterrupt
                        print('Invalid. Choose two cards')
            case 'random':
                # Garunteed to be random
                return [1,2]
            case 'fit':
                return optimal_discard(deal,is_dealer=is_dealer)
            case _:
                raise Exception(f'No such strategy {self.strategy}')

    def play(self, hand, inplay) -> int:
        '''
        Phase 2 of cribbage. Choose card from hand to play
        '''
        # say 'go' if no cards can be played
        max_play_value = score_value['max_tally'] - sum_cards(inplay)
        iplayable = list(filter(lambda k: card_value[hand[k].Value] <= max_play_value, range(len(hand))))

        # choose card to play
        match self.strategy:
            case 'human':
                sel = 0
                for k,card in enumerate(hand):
                    if k in iplayable:
                        print(sel, card)
                        sel += 1
                    else:
                        print(' ', card)

                while True:
                    try:
                        if not iplayable:
                            input('Enter: say "GO"')
                            return 'GO'
                        return int(input('Choose play: '))
                    except Exception as err:
                        if C == 'q':
                            raise KeyboardInterrupt
                        print('Invalid. Choose one cards')
            case 'random':
                if not iplayable:  # no playable cards, say go
                    return 'GO'
                return iplayable[-1]  # Discard highest playable value # TODO better strategy for computer
            case 'fit':
                return optimal_play(hand, inplay)
            case _:
                raise Exception(f'No such strategy {self.strategy}')

        

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
