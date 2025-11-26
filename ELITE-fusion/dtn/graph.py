# Minimal junction-graph helper for DTN training.
# We synthesize a small grid road network with Euclidean coordinates,
# and expose it via the same "it_pos" / "adjacents_comb" shape that the
# rest of the project expects.

from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple
import math

Coord = Tuple[float, float]

def make_grid(n_rows: int = 4, n_cols: int = 4, spacing: float = 250.0) -> Tuple[Dict[int, Coord], Dict[int, List[int]]]:
    """
    Create a n_rows x n_cols grid of junctions.
    Node ids are 0..N-1 row-major, coordinates are (col*spacing, row*spacing).
    Adjacency is 4-neighbor where applicable.
    Returns:
      it_pos: Ordered mapping id -> (x,y)
      adj: Ordered mapping id -> [neighbor ids] in ascending id order
    """
    it_pos: "OrderedDict[int, Coord]" = OrderedDict()
    adj: "OrderedDict[int, List[int]]" = OrderedDict()
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            it_pos[idx] = (c * spacing, r * spacing)
            idx += 1
    total = n_rows * n_cols
    for i in range(total):
        r, c = divmod(i, n_cols)
        nbrs: List[int] = []
        if r > 0:
            nbrs.append(i - n_cols)
        if r < n_rows - 1:
            nbrs.append(i + n_cols)
        if c > 0:
            nbrs.append(i - 1)
        if c < n_cols - 1:
            nbrs.append(i + 1)
        adj[i] = sorted(nbrs)
    return it_pos, adj

def euclid(a: Coord, b: Coord) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)

