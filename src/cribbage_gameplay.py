from typing import Set
from itertools import combinations, chain

# cribbage specific rules
card_value = [1,2,3,4,5,6,7,8,9,10,10,10,10] # maps to `Value`

# Counter and score for each score type.
breakdown_counter = \
    {'zero': 0, 
     'fifteen': 0, 
     'pair': 0, 
     'run3': 0, 
     'run4': 0, 
     'run5': 0, 
     'flush4': 0, 
     'flush5': 0, 
     'nibs': 0
    }
score_value = \
    {'zero': 0,     # count
     'fifteen': 2,  # count
     'pair': 2,     # count
     'run': 1,      # multiplier
     'flush4': 4,   # count
     'flush5': 5,   # count
     'straight_flush': 2,   # multiplier
     'nibs': 1,     # count
     'nobs': 2      # count
    }

def breakdown_counter_empty ():
    return breakdown_counter.copy()

class CribCountException(Exception):
    pass

def isnibs(hand):
    for c in hand[1:]:
        if c.Value == 10 and c.Suit == hand[0].Suit:
            return True
    return False
def isflush(hand):
    # all cards are the same suit
    return len(set(map(lambda c: c.Suit, hand)))==1
def isflush_in_hand(hand):
    return len(set(map(lambda c: c.Suit, hand[:4])))==1
def isrun(sub):
    return  all([k==(c.Value-sub[0].Value) for k,c in enumerate(sub)])
def isfifteen(sub):
    return sum([card_value[c.Value] for c in sub]) == 15
def ispair(sub):
    return len(sub)==2 and sub[0].Value==sub[1].Value

def cribscore(hand):
    breakdown = breakdown_counter_empty()
    score = 0

    # jack
    if isnibs(hand):
        breakdown['nibs'] += 1
        score += score_value['nibs']
        
    # flush
    if isflush(hand):
        breakdown['flush5'] += 1
        score += score_value['flush5']
    elif isflush_in_hand(hand):
        breakdown['flush4'] += 1
        score += score_value['flush4']
    
    # Run thorugh all card combos
    combs = chain(*[combinations(hand,k) for k in range(len(hand),1,-1)])
    minrunlen = 3
    for sub in combs:
        sublen = len(sub)
        sub = sorted(sub, key=lambda c: c.Value)

        # fifteens
        if isfifteen(sub):
            breakdown['fifteen'] += 1
            score += score_value['fifteen']
        
        # run
        if sublen>=minrunlen and isrun(sub):
            breakdown['run'+str(sublen)] += 1
            score += score_value['run'] * sublen
            minrunlen = sublen
            
        # pair
        if len(sub)==2 and ispair(sub):
            breakdown['pair'] += 1
            score += 2
        
        #print(sub)

    if score==0:
        breakdown['zero'] += 1
             
    return score, breakdown
def chantscore(bd):
    chant = []
    scorecalc = 2 * bd['fifteen'] + \
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
        # chant.append('Nineteen!') # only in some bars
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

    if not scorecalc == running:
        raise CribCountException('Miscount during cribchant')
    return chant
