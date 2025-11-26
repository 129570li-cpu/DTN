#!/usr/bin/env python3
"""
Run a minimal DTN training to produce four single-target Q-tables,
then demonstrate fusion via existing Routing_table.
This script does not depend on external XML/TCL assets; it synthesizes
an N x M grid topology and writes CSV files under ./dtn_out/.
"""
import os
import sys

# Make relative imports work when executed directly
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
sys.path.insert(0, ROOT)

from dtn.graph import make_grid
from dtn.train import train_and_export
import Global_Par as Gp
from Routing_table import Routing_Table

def main():
    out_dir = os.path.join(ROOT, "dtn_out")
    paths = train_and_export(out_dir, n_rows=4, n_cols=4, episodes=4000, seed=1)
    # Inject the synthesized topology into Global_Par for shaping DataFrames
    it_pos, adj = make_grid(n_rows=4, n_cols=4, spacing=250.0)
    Gp.it_pos = it_pos
    Gp.adjacents_comb = adj
    # Point the file names to the newly exported CSVs (no need to change Gp.n)
    Gp.file_pdr = paths["pdr"]
    Gp.file_ad  = paths["ad"]
    Gp.file_hc  = paths["hc"]
    Gp.file_rc  = paths["rc"]
    # Build routing tables and fuse
    rt = Routing_Table()
    rt.preprocessing()
    rt.fusion_weight()
    rt.fusion_fuzzy()
    print("Fusion complete. Tables available as rt.table_BP / rt.table_HRF / rt.table_LDF / rt.table_LBF")

if __name__ == "__main__":
    main()

