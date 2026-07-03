from abc import ABC, abstractmethod
import copy
from colorama import Style
import numpy as np
from Solver import Solver
from scipy.ndimage import label


class Block:
    def __init__(self, positions, rgb, letter, terminal_color):
        positions = np.trim_zeros(positions)
        self.width, self.height = positions.shape
        self.rgb = rgb
        self.terminal_color = terminal_color
        self.letter = letter
        self.is_connected(positions)
        self.positions = positions
        self.count = int(np.sum(positions))
        assert self.count > 0, "Block must have at least one cell."
            
    def is_connected(self, positions): 
        label_structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
        n_components = label(positions, structure=label_structure)[1]
        assert n_components == 1, "Block is not connected."


class Game(ABC):
    """Shared puzzle bookkeeping. Subclasses define the geometry."""
    def __init__(self, blocks, name, grid=None, shape=None):
        assert -1 not in blocks, "-1 is reserved for empty cells and can't be a block index"
        self.shape = shape if shape is not None else tuple(np.asarray(grid).shape)
        self.blocks = blocks
        self.name = name
        self.is_valid()

        placements = {idx: self._compute_placement_indices(b) for idx, b in blocks.items()}
        block_counts = {idx: b.count for idx, b in blocks.items()}
        self.solver = Solver(np.full(self.shape, -1, dtype=int), placements,
                            block_counts, label_structure=self._label_structure())

        if grid is not None:
            assert grid.shape == self.shape, "grid shape mismatch"
            if grid.dtype != int:
                grid = self.grid_from_letters(grid)
            self.solver.initialize_grid(grid)

    @property
    def grid(self):
        return self.solver.grid
    
    def grid_from_letters(self, letter_rows, empty=' '):
        """letter_rows: nested sequence of single-char letters (or `empty`), shape == self.shape."""
        letter_to_idx = {b.letter: idx for idx, b in self.blocks.items()}
        arr = np.asarray(letter_rows)
        idx_grid = np.full(arr.shape, -1, dtype=int)
        for letter, idx in letter_to_idx.items():
            idx_grid[arr == letter] = idx
        unknown = set(np.unique(arr)) - set(letter_to_idx) - {empty}
        assert not unknown, f"unknown letter(s) in grid: {unknown}"
        return idx_grid

    # ---- hooks every dimension must implement ----
    @abstractmethod
    def _compute_placement_indices(self, block): ...

    @abstractmethod
    def is_valid(self): ...

    @abstractmethod
    def _label_structure(self): ...

    @abstractmethod
    def print_to_terminal(self, grid): ...   # e.g. print_to_terminal for 2D, whatever for 3D

    # ---- generic glue, delegates to solver ----
    def solve(self, disp: bool = False, mode: int = 0, seed: int = None):
        if disp:
            print(f"Solving {self.name} with mode {mode}...")
            self.print_to_terminal(self.grid)
        for sol_idx,sol in enumerate(self.solver.solve(mode=mode, seed=seed)):
            if disp:
                print(f"  Solution {sol_idx+1}:")
                self.print_to_terminal(sol)
            yield sol

    def copy(self, rename=None):
        clone = copy.deepcopy(self)
        if rename is not None:
            clone.name = rename
        return clone
    
class FlatGame(Game):
    def __init__(self, width: int, height: int, blocks: dict, name: str, grid: np.ndarray = None):
        self.width, self.height = width, height
        super().__init__(blocks, name, grid, (width, height))

    def _compute_placement_indices(self, block: Block):
        """
        Every unique rotation/flip of self.positions, placed everywhere it
        fits in a (width, height) grid, stacked into ONE array of shape
        (total_placements, self.count) -- flat row-major indices into the grid.
        Assumes every symmetry has at least one valid placement.
        """
        W, H = block.width, block.height  # block's own untransformed dims

        p = block.positions
        syms = (
            p, np.rot90(p, 1), np.rot90(p, 2), np.rot90(p, 3),
            np.flip(p, 0), np.rot90(np.flip(p, 0), 1),
            np.rot90(np.flip(p, 0), 2), np.rot90(np.flip(p, 0), 3),
        )
        coords = np.array([np.argwhere(s) for s in syms])  # (8, count, 2)
        _, idx = np.unique(coords.reshape(8, -1), axis=0, return_index=True)
        order = np.sort(idx)
        coords = coords[order]                              # (n_sym, count, 2)
        count = coords.shape[1]

        # k=1,3 (odd) transpose the bounding box; k=0,2 don't -- so the
        # extent of each symmetry is known directly from W,H, no .max() needed.
        odd = (order % 2 == 1)
        row_extent = np.where(odd, H, W)   # (n_sym,)
        col_extent = np.where(odd, W, H)   # (n_sym,)

        n_starts_row = self.width - row_extent + 1
        n_starts_col = self.height - col_extent + 1
        n_place = n_starts_row * n_starts_col   # (n_sym,) placements per symmetry

        result = np.empty((n_place.sum(), count), dtype=np.int64)

        offset = 0
        for sym_i in range(coords.shape[0]):
            row_starts = np.arange(n_starts_row[sym_i])
            col_starts = np.arange(n_starts_col[sym_i])
            cell_rows = coords[sym_i, :, 0]
            cell_cols = coords[sym_i, :, 1]

            rr = np.repeat(row_starts, col_starts.size)[:, None] + cell_rows
            cc = np.tile(col_starts, row_starts.size)[:, None] + cell_cols

            n = n_place[sym_i]
            result[offset:offset + n] = rr * self.height + cc
            offset += n

        return result

    def is_valid(self):
        # TODO: does this work for initialized puzzle
        total_nr_dots = np.sum([b.count for b in self.blocks.values()])
        assert total_nr_dots == self.width*self.height
        
        for b in self.blocks.values():
            # TODO: placements function requires the piece to fit in both directions. fix that.
            assert (b.width<self.width and b.height<self.height) and (b.width<self.height or b.height<self.width)

    def _label_structure(self):
        return np.array([[0,1,0],[1,1,1],[0,1,0]])

    def print_to_terminal(self,grid:np.ndarray,print_letters:bool=True):
        # since terminal doesnt support RGB colors, we just arbitrarily assign a color to each block
        assert grid.shape == (self.width, self.height), "Grid shape does not match game dimensions"

        print('┏'+2*self.width*'━'+'┓')
        for row in grid.T:
            print('┃',end='')
            for cell in row:
                if cell!=-1:
                    if print_letters:
                        txt = self.blocks[cell].letter + ' '
                    else:
                        txt = '  '
                    print(self.blocks[cell].terminal_color + txt + Style.RESET_ALL,end='')
                else:
                    print('  ',end='')
            
            print('┃')
        print('┗'+2*self.width*'━'+'┛')

    def write_to_html(self, filename):
        # Pre-calculate mapping
        color_map = {id: f"rgb({b.rgb[0]}, {b.rgb[1]}, {b.rgb[2]})" 
                    for id, b in self.blocks.items()}
        
        with open(filename, "w") as f:
            f.write("<html><style>")
            f.write("body { background-color: #2d2d2d; color: white; font-family: sans-serif; }")
            f.write(".cell { width: 20px; height: 20px; display: inline-block; box-sizing: border-box; }")
            f.write("</style><body>")
            
            # Add the title
            f.write(f"<h2>{self.name}</h2>")
            
            f.write("<div style='line-height: 0;'>")
            
            for row in self.grid.T:
                for cell_id in row:
                    bg_color = color_map.get(cell_id, "transparent")
                    f.write(f"<div class='cell' style='background-color: {bg_color};'></div>")
                f.write("<br>")
                
            f.write("</div></body></html>")