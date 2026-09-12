from classes import FlatBoard, PyramidBoard, Game
from colorama import Back, Fore, Style
import numpy as np

from classes import Block, BlockCollection, FlatBoard, PyramidBoard

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
                                     [1, 0, 0, 0]],dtype=bool),
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

                                     

BLOCKS = BlockCollection(
    RED, ORANGE, YELLOW,
    DARKBLUE, DARKGREEN, DARKPINK,
    GRAY, LIGHTBLUE, LIGHTGREEN, LIGHTPINK,
    WHITE, PURPLE,
)


MainBoard = FlatBoard(width=11, height=5)
BackBoard = PyramidBoard(5,5)

BackBoard2 = FlatBoard(cells=np.array([ [0, 1, 1, 1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 0, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 0, 1, 1 ,1 ,1 ,1 ,1, 1],
                                        [0, 0, 0, 0, 1, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=bool))

G1 = Game(BLOCKS, BackBoard)
G2 = Game(BLOCKS, MainBoard)


for block_idx,block in G1.blocks.items():
    print(f"block {block.letter} has {len(G1.placements[block_idx])} placements in the pyramid")
    print(f"block {block.letter} has {len(G2.placements[block_idx])} placements in the main board")
    print()
