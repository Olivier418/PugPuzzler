from ._utils import _assert_unique, _shared
from .blocks import BlockCollection
from .puzzle import Puzzle, PuzzleBook, Setup


class Game:
    """Container for a game package, holding puzzle books, standalone loose

    puzzles, and board setups.
    """

    def __init__(
        self,
        books: list[PuzzleBook] = None,
        puzzles: list[Puzzle] = None,
        setups: dict[str, Setup] = None,
        name: str = None,
    ):
        books = books or []
        puzzles = puzzles or []

        _assert_unique(books, lambda b: b.name, "puzzlebook name")
        _assert_unique(puzzles, lambda p: p.name, "puzzle name")

        self.name = name
        self.books: dict[str, PuzzleBook] = {b.name: b for b in books}
        self.puzzles: dict[str, Puzzle] = {p.name: p for p in puzzles}
        self.setups: dict[str, Setup] = setups or {}

        all_items = list(books) + list(puzzles)
        self.blocks: BlockCollection = (
            _shared(all_items, lambda x: x.blocks, "BlockCollection") if all_items else None
        )