import random
from src.cribbage_gameplay import CribbageMatch
from src.cribbage import CribAgent

random.seed

def inputs(text, type_in=str):
    while True:
        try:
            return type_in(input())
        except Exception as e:
            print('Invalid. Try again.')

print('LETS PLAY CRIBBAGE')
print('')
print('Who is playing?')
print('1. human v human')
print('2. human v computer')
print('3. computer v computer')

match inputs('select: ', int):
    case 1:
        agent1 = CribAgent('human', input('Player 1 name: '))
        agent2 = CribAgent('human', input('Player 2 name: '))
    case 2:
        agent1 = CribAgent('human', input('Player 1 name: '))
        agent2 = CribAgent('fit', 'Frank')
    case 3:
        agent1 = CribAgent('fit', 'Terry')
        agent2 = CribAgent('fit', 'Frank')
    case _:
        raise ValueError('I don\'t know that game')

match = CribbageMatch(agent1, agent2, endscore = 120, verbose = True)

match.playmatch()