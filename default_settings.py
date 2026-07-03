from colorama import Back, Fore, Style
import numpy as np

from classes import FlatGame, Block

ORANGE = Block(positions=np.array([[1, 0, 0], 
                                   [1, 1, 1]],dtype=bool),
               rgb=(255, 165, 0),         
               letter='A',
               terminal_color = Back.LIGHTRED_EX + Fore.BLACK)
RED = Block(positions=np.array([[1, 1, 0], 
                                [1, 1, 1]],dtype=bool),
            rgb=(255, 0, 0), 
            letter='B',
            terminal_color = Back.RED + Fore.BLACK)
DARKBLUE = Block(positions=np.array([[1, 1, 1, 1],
                                     [1, 0, 0, 0]]),
                rgb = (0, 0, 255),
                letter='C',
                terminal_color = Back.BLUE + Fore.BLACK)
LIGHTPINK = Block(positions=np.array([[1, 1, 1, 1],
                                      [0, 1, 0, 0]],dtype=bool),
                rgb=(255, 192, 203),
                letter='D',
                  terminal_color= Back.LIGHTMAGENTA_EX + Style.DIM +Fore.WHITE) 
DARKGREEN = Block(positions=np.array([[1, 1, 1, 0],
                                      [0, 0, 1, 1]],dtype=bool),
                  rgb=(0, 100, 0),
                  letter='E',
                  terminal_color = Back.GREEN + Fore.WHITE)
WHITE = Block(positions=np.array([[1, 1],
                                  [0, 1]],dtype=bool),
                rgb=(255, 255, 255),
                letter='F',
                terminal_color = Back.WHITE + Fore.BLACK)
LIGHTBLUE = Block(positions=np.array([[1, 1, 1], 
                                      [1, 0, 0],
                                      [1, 0, 0]],dtype=bool),
                rgb=(173, 216, 230),
                letter='G',
                terminal_color = Back.CYAN + Fore.BLACK) 
DARKPINK = Block(positions=np.array([[1, 1, 0],
                                     [0, 1, 1],
                                     [0, 0, 1]],dtype=bool),  
                 rgb=(255, 192, 203),
                 letter='H',
                 terminal_color = Back.LIGHTMAGENTA_EX + Style.BRIGHT +Fore.WHITE)
YELLOW = Block(positions=np.array([[1, 0, 1], 
                                   [1, 1, 1]],dtype=bool),
                rgb=(255, 255, 0),
                letter='I',
                terminal_color = Back.YELLOW + Fore.BLACK)
PURPLE = Block(positions=np.array([[1, 1, 1, 1]],dtype=bool),
                rgb=(128, 0, 128),
                letter='J',
                terminal_color = Back.MAGENTA + Fore.BLACK)
LIGHTGREEN = Block(positions=np.array([[1, 1],
                                       [1, 1]],dtype=bool),
                   rgb=(0, 255, 0),
                   letter='K',
                   terminal_color = Back.LIGHTGREEN_EX + Fore.BLACK)
GRAY = Block(positions=np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],dtype=bool),
             rgb=(128, 128, 128),
             letter='L',
             terminal_color = Back.BLACK + Fore.WHITE)

                                     
def make_blocks(block_list):
    blocks = dict(enumerate(block_list))
    letters = [b.letter for b in blocks.values()]
    assert len(letters) == len(set(letters)), \
        f"duplicate block letters: {[l for l in letters if letters.count(l) > 1]}"
    return blocks

BLOCKS = make_blocks([
    RED, ORANGE, YELLOW,
    DARKBLUE, DARKGREEN, DARKPINK,
    GRAY, LIGHTBLUE, LIGHTGREEN, LIGHTPINK,
    WHITE, PURPLE,
])

WIDTH,HEIGHT = 11, 5
EMPTY = FlatGame(WIDTH, HEIGHT, BLOCKS, name="EMPTY")