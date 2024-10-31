import random
from itertools import combinations, chain

if __name__ == "__main__":
    from playing_cards import full_deck, discard_from_hand
else:
    from .playing_cards import full_deck, discard_from_hand

# cribbage specific rules
card_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # maps to `Value`

# Counter and score for each score type.
breakdown_counter = {
    'zero': 0,
    'fifteen': 0,
    'pair': 0,
    'run3': 0,
    'run4': 0,
    'run5': 0,
    'flush4': 0,
    'flush5': 0,
    'nibs': 0
}
score_value = {
    'zero': 0,     # count
    'fifteen': 2,  # count
    'pair': 2,     # count
    'run': 1,      # multiplier
    'flush4': 4,   # count
    'flush5': 5,   # count
    'straight_flush': 2,   # multiplier
    'nibs': 1,     # count
    'nobs': 2,     # count
    'last card': 1,# play
    '31': 1,       # play
    'go': 1,       # play
    'max_tally': 31 # rules
}


def breakdown_counter_empty():
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
    return len(set(map(lambda c: c.Suit, hand))) == 1


def isflush_in_hand(hand):
    return len(set(map(lambda c: c.Suit, hand[1:]))) == 1


def isrun(sub):
    return all([k == (c.Value-sub[0].Value) for k, c in enumerate(sub)])


def isfifteen(sub):
    return sum([card_value[c.Value] for c in sub]) == 15


def ispair(sub):
    return len(sub) == 2 and sub[0].Value == sub[1].Value


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
    combs = chain(*[combinations(hand, k) for k in range(len(hand), 1, -1)])
    minrunlen = 3
    for sub in combs:
        sublen = len(sub)
        sub = sorted(sub, key=lambda c: c.Value)

        # fifteens
        if isfifteen(sub):
            breakdown['fifteen'] += 1
            score += score_value['fifteen']

        # run
        if sublen >= minrunlen and isrun(sub):
            breakdown['run'+str(sublen)] += 1
            score += score_value['run'] * sublen
            minrunlen = sublen

        # pair
        if len(sub) == 2 and ispair(sub):
            breakdown['pair'] += 1
            score += 2

        # print(sub)

    if score == 0:
        breakdown['zero'] += 1

    return score, breakdown


def chantscore(bd):
    chant = []
    scorecalc = \
        2 * bd['fifteen'] + \
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
        # assert scorecalc == running, CribCountException('Zero marked but points scored!')
        # return chant, scorecalc# save time

    if bd['fifteen']:
        chant.append('Fifteen ' + ' '.join([str(k) for k in range(2, 2*bd['fifteen'] + 1, 2)]))
        running += 2 * bd['fifteen']

    counted_pairs = False
    if bd['run3'] > 1:  # double(s) run
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
    elif bd['run4'] > 1:
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
        if bd['pair'] == 1:
            pair_phrase = 'Pair'
        elif bd['pair'] > 2:
            pair_phrase = 'Two pair'
        elif bd['pair'] == 3:
            pair_phrase = 'Tripsies'
        elif bd['pair'] == 4:
            pair_phrase = 'Quadsies'
        else:
            pair_phrase = 'Pairs'
            
        chant.append(f'{pair_phrase} for {running}')
    
    if bd['flush4']:
        running += 4
        chant.append(f'Flush of 4 for {running}')
    elif bd['flush5']:
        running += 5
        chant.append(f'Flush of 5 for {running}')
    
    if bd['nibs']:
        running += 1
        chant.append(f'His nibs for {running}')

    if len(chant) > 1:
        chant[-1] = 'and ' + chant[-1]

    chant = '\n'.join(chant)

    if not scorecalc == running:
        raise CribCountException('Miscount during cribchant')
    return chant

class CribbageGameEnded(Exception):
    pass

class CribbageMatch:
    scoreboard = {True: 0, False: 0}
    has_winner: bool = False
    winner: bool = None
    dealer: bool

    verbose: bool

    def __init__(self, agent1, agent2, endscore=120, verbose=False):
        self.agents = {True: agent1, False: agent2}
        self.endscore = endscore

        self.dealer = random.choice([True, False])

        self.verbose = verbose

    @property
    def player(self) -> bool:
        return not self.dealer
    
    @property
    def dealer_name(self) -> str:
        return self.agents[self.dealer].name

    @property
    def player_name(self) -> str:
        return self.agents[self.player].name

    def print(self, *message):
        if self.verbose:
            print(*message)
    
    def update_score(self, points, player):
        self.scoreboard[player] += points
        self.check_for_winner()

    def do_the_deal(self):
        '''
        Phase 1 of cribbage round
        '''

        # deal the cards
        deck_shuffled = full_deck(do_shuffle=True)
        dealt = {self.player: deck_shuffled[0:6], 
                 self.dealer: deck_shuffled[6:12],
                 'cut': deck_shuffled[-1]}

        self.print(f'Player1: {self.dealer_name} (dealer)')
        self.print(f'Player2: {self.player_name}')
        self.print('Deal: ', dealt)  #DEBUG

        # get player discards
        players = {self.player, self.dealer}
        idiscard = { p: self.agents[p].discard(dealt[p]) for p in players}

        # distribute hands
        hands = dict()
        hands[self.player], player_discard = discard_from_hand(dealt[self.player],idiscard[self.player])
        hands[self.dealer], dealer_discard = discard_from_hand(dealt[self.dealer],idiscard[self.dealer])
        hands['crib'] = player_discard + dealer_discard

        self.print('Hands: ', hands)

        self.print(f'The cut: {dealt['cut']}')
        if dealt['cut'].value_symb == 'J':
            self.print('HIS NOBS')
            self.update_score(score_value['nobs'], self.dealer)

        return dealt, hands

    def do_the_play(self, hands):
        pass
        # '''
        # Phase 2 of cribbage round
        # '''
        # turn = self.player
        # next_turn = None
        # tally = 0

        # says_go = {self.dealer: False, self.player: False}
        # inhand = {p: hands[p].copy() for p in [True,False]}  # Copy of hands to discard from
        # inplay = []  # run of played cards

        # while True:
        #     # get player discard
        #     played = self.agents[turn].play(inhand[turn],inplay)

        #     if played == 'go':  # GO
        #         print(f'player {self.agent[turn].name} says go')
        #         says_go[turn] = True
        #         continue
        #     elif not played: # Out of Cards
        #         continue

        #     assert isinstance(played, int), f'Player must play card, not {played}'

        #     # Add card to play stack
        #     inplay.append(played)

        #     # score points from play
        #     self.update_score(turn, self.score_play(inplay) )

        #     # end play if all cards are played
        #     if hands[self.player] or hands[self.dealer]:
        #         self.update_score(turn, score_value['last card'])
        #         break

        #     # end of round
        #     if all(says_go.values()) or (is31:=sum(inplay)==score_value['max_tally']):
        #         self.update_score(turn, score_value['go'])
        #         self.update_score(turn, is31)

        #     # switch turns
        #     turn = not turn

    def score_hand(self,playerhands,cut):

        self.print(f'Player {self.player_name} counts hand..')
        self.print(whole_hand)
        score, breakdown = cribscore(whole_hand)
        self.print(chantscore(breakdown))
        self.update_score(score, self.player)
        self.print('')

    def do_the_count(self, hands, dealt):
        '''
        Phase 3 of cribbage round
        Count player hand, dealer hand and dealer crib
        Check for win after each count
        ''' 
        cut = dealt['cut']

        # Player hand
        self.print(f'Player {self.player_name} counts hand..')
        self.print([cut], ' ', hands[self.player])
        player_score, player_breakdown = cribscore([cut] + hands[self.player])
        self.print(chantscore(player_breakdown))
        self.update_score(player_score, self.player)
        self.print('')

        # dealer hand
        self.print(f'Dealer {self.dealer_name} counts hand ..')
        self.print([cut], ' ', hands[self.dealer])
        dealer_score, dealer_breakdown = cribscore([cut] + hands[self.dealer])
        self.print(chantscore(dealer_breakdown))
        self.update_score(dealer_score, self.dealer)
        self.print('')

        # dealer crib
        self.print(f'Dealer {self.dealer_name} counts crib ..')
        self.print([cut], ' ', hands['crib'])
        crib_score, crib_breakdown = cribscore([cut] + hands['crib'])
        self.print(chantscore(crib_breakdown))
        self.update_score(crib_score, self.dealer)
        self.print('')

    def playround(self):
        # Phase 1: Deal and discard
        dealt, hands = self.do_the_deal()

        # Phase 2: the play
        self.do_the_play(hands)

        # Phase 3; the count
        self.do_the_count(hands,dealt)

        self.dealer = not self.dealer

    def playmatch(self):
        round_count = 0
        looplimit = self.endscore/3

        self.print('Begin match')

        while not self.has_winner and round_count < looplimit:
            round_count += 1
            print(f'Round {round_count}')
            try: 
                self.playround()
                self.print(self.scoreboard)
            except CribbageGameEnded as e:
                if not self.has_winner:
                    print('somehow, no winner')
        
        self.congratulate()

        if round_count >= looplimit:
            raise CribbageGameEnded('Cribbage match exited at loop limit. This is suspicious')

    def check_for_winner(self):
        winning = { p: self.scoreboard[p] >= self.endscore for p in [True, False]}

        self.has_winner = any(winning.values())

        if all(winning.values()):
            # Somehow, both players passed the winning mark. This shouldn't happen.
            self.winner = None
            raise CribbageGameEnded('Both players won')
        elif winning[self.dealer]:
            self.winner = self.dealer
        elif winning[self.player]:
            self.winner = self.player
        else:
            self.winner = None
            return
        
        raise CribbageGameEnded('Winner!')

    def congratulate(self):
        print(f'Winner! {self.agents[self.winner].name}')
        print(f'Player1:{self.agents[True].name}: {self.scoreboard[True]}')
        print(f'Player2:{self.agents[False].name}: {self.scoreboard[False]}')

if __name__ == "__main__":
    print(full_deck())