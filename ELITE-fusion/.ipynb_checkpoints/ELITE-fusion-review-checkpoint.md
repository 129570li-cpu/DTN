ELITE-fusion Engineering Review (for ELITE/IDT-SDVN)
====================================================

Scope
- Repository path: `ELITE-fusion`
- Goal: assess how far this code reproduces the paper “ELITE: An Intelligent Digital Twin-Based Hierarchical Routing Scheme for Softwarized Vehicular Networks” (IEEE TMC 2023), and record gaps and next steps.

High‑Level Findings
- Implemented (partial)
  - Road-junction level hierarchical routing with controller-side path planning and vehicle-side greedy forwarding.
  - Experience fusion (EFG): both weight-based fusion (BP) and fuzzy-logic fusion generating HRF/LDF/LBF routing tables from 4 single-target tables (PRR/AD/HC/RC) after normalization.
  - Basic SDVN controller object managing node info, neighbor sets, and invoking table fusion.
- Missing or incomplete (key)
  - SPL (parallel Q-learning) is not implemented; single-target “Q tables” are loaded from CSV instead.
  - APN (adaptive policy deployment via state–action table) is missing; strategy selection is a manual global flag.
  - RSU-assisted variant (ELITE-RSU) and baseline schemes (QCR/IV2XQ/IGCR) are not implemented.
  - Simulation assets (map XML and mobility TCL) are absent, so the current code cannot run end-to-end.

Paper → Code Mapping
- Junction-based path planning
  - Controller computes a junction sequence from source to destination by consulting a selected routing table.
  - Code: `ELITE-fusion/SDVN_Controller.py:78` `calculate_area_path(...)` selects the next junction by the maximum table entry (with loop detection) and returns an area/junction path.
- Vehicle hop-by-hop forwarding (road-based greedy with void handling)
  - Code: `ELITE-fusion/Node.py:237` `intra_area(...)` and `ELITE-fusion/Node.py:350` `find_next(...)`; geometric helpers also in `ELITE-fusion/GPSR.py:1`.
- Experience Fusion (EFG)
  - Normalization of four learned tables: PRR/AD/HC/RC.
    - Code: `ELITE-fusion/Routing_table.py:112-167` `preprocessing(...)`.
  - Weight-based fusion (BP).
    - Code: `ELITE-fusion/Routing_table.py:171-177` `fusion_weight(...)`.
  - Fuzzy-logic fusion for HRF/LDF/LBF; triangular membership, Min–Max inference, CoG defuzzification.
    - Code: `ELITE-fusion/Routing_table.py:185-274` `fusion_fuzzy(...)`.
    - Fuzzy internals: `ELITE-fusion/fuzzy/FuzzyRules.py:15` `class input`, `ELITE-fusion/fuzzy/FuzzyRules.py:53` `class rule`, `ELITE-fusion/fuzzy/FuzzyRules.py:294` `defuzzy(...)`.

What’s Missing vs. Paper
- SPL: Parallel Q-learning in DTNs
  - No online/parallel agent training, no three-case junction selection (exploitation/greedy/exploration), no reward decomposition (RPDR/RAD/RHC/RRC) nor table updates.
  - `ELITE-fusion/Virtual_agents.py:69-71` `learning()` is empty; `Virtual_agents.table_config` currently calls `get_matrix()` without parameter (bug; should pass a file name).
- APN: State–Action policy deployment and updating
  - Paper uses a state–action table where state = message type × load level, action ∈ {GHRF, GLDF, GLBF}, and updates P(state, action) after each delivery.
  - In code, strategy is manually chosen via `Gp.tag`, and report handling is a stub:
    - Strategy choice: `ELITE-fusion/SDVN_Controller.py:79-86` selects `table_HRF/LDF/LBF/BP` by `Gp.tag`.
    - Report update: `ELITE-fusion/SDVN_Controller.py:146-151` `resolve_report()` is empty (no state–action update).
- RSU/Variants/Baselines
  - ELITE-RSU (junction RSU-assisted), QCR, IV2XQ, IGCR are not present.
- Metrics/CO/RO feedback loop
  - While containers and some metric utilities exist (`ELITE-fusion/Metrics.py`), the on-delivery report → reward → table update loop is not wired.
- Simulation assets and reproducibility
  - Required files are missing:
    - Map: `de_intersection.xml`, `de_edges.xml` referenced at `ELITE-fusion/traffic.py:32` and `ELITE-fusion/traffic.py:47`.
    - Mobility: `de-*.mobility.tcl` referenced at `ELITE-fusion/test.py:31,36,41,46,51,66`.
  - Without these, even table construction fails because topology (`Gp.it_pos`, `Gp.adjacents_comb`) is empty, making DataFrame shapes inconsistent.

Observed Runtime Blocker (example)
- Reproduced a failure when trying to instantiate `Routing_Table` with the provided CSVs (no map loaded):
  - `ValueError: 0 columns passed, passed data had 12 columns` arising from `pd.DataFrame(matrix, index=index_, columns=column_)`.
  - Root cause: `column_` is empty because topology is not initialized (needs `Gp.it_pos`, `Gp.adjacents_comb` from map).
  - Code path: `ELITE-fusion/Routing_table.py:73-86` `get_table(...)` → `table_config(...)`.

Code Quality / Minor Issues
- `ELITE-fusion/Virtual_agents.py:32-53`
  - `table_config(self)` calls `self.get_matrix()` without the required `table_name` argument; this will raise a `TypeError`.
- `ELITE-fusion/SDVN_Controller.py:146-151`
  - `resolve_report()` is a placeholder; should parse reports and update policy selection (state–action).
- CSV/Q-table dimensional assumptions
  - The CSVs (e.g., `table_record_0_pdr.csv`) have fixed numeric shapes; they must match the derived index from `Gp.it_pos` and `Gp.adjacents_comb`, otherwise DataFrame construction breaks.

What Works (quick)
- Fusion pipeline from 4 “learned” tables to BP/HRF/LDF/LBF is complete, including:
  - Per-destination/per-current/per-neighbor 3D table shaping as Pandas DataFrames with multi-index rows and destination columns.
  - Alternative membership thresholds via maxima (reflecting paper’s use of mv to set triangle intersections).
- Controller path planning and vehicle forwarding integrate with the fused tables once topology and node states are available.

How To Complete Toward Paper Fidelity
1) Make it runnable
   - Provide/commit the required map XML (`de_intersection.xml`, `de_edges.xml`) and mobility TCL traces (`de-*.mobility.tcl`), or add a small synthetic grid map and random mobility generator to avoid external files.
   - Ensure `Gp.it_pos`, `Gp.adjacents_comb`, and distance dictionaries are initialized before building routing tables.
2) Implement SPL (parallel Q-learning)
   - In `Virtual_agents.learning()`:
     - Generate virtual transmissions in DTNs; implement three-path selection modes (exploit/greedy/random) per the paper.
     - Apply reward definitions RPDR/RAD/RHC/RRC and back-propagate along successful paths to update the corresponding Q tables.
     - Serialize learned Q tables to CSV for the existing fusion pipeline.
3) Implement APN (state–action)
   - Add a `StateActionTable` structure keyed by (message_type, load_level) → {GHRF, GLDF, GLBF}.
   - Replace `Gp.tag` selection with querying the state–action table; compute path load (paper’s formula) using `Gbas` to determine state.
   - On report (`FlowReport`), compute reward `R0` and update `P(state, action)` (additive/one-step as in the paper).
4) Optional variants and baselines
   - ELITE-RSU: model RSUs at junctions; on arrival to a junction, forward to RSU which selects the next vehicle.
   - Baselines (QCR/IV2XQ/IGCR) for comparison figures.
5) Evaluation scripts
   - Aggregate PDR, AD, PL, RO, CO; generate plots comparable to Fig. 6/Fig. 7 in the paper.

Quick Code Pointers (for development)
- Controller path planning and fusion
  - `ELITE-fusion/SDVN_Controller.py:78` `calculate_area_path(...)`
  - `ELITE-fusion/SDVN_Controller.py:155-159` `table_fusion(...)`
  - `ELITE-fusion/Routing_table.py:112-167` normalization
  - `ELITE-fusion/Routing_table.py:171-177` BP fusion
  - `ELITE-fusion/Routing_table.py:185-274` HRF/LDF/LBF fusion
- Fuzzy logic internals
  - `ELITE-fusion/fuzzy/FuzzyRules.py:15` input → fuzzy sets
  - `ELITE-fusion/fuzzy/FuzzyRules.py:53` rule aggregation (Min–Max)
  - `ELITE-fusion/fuzzy/FuzzyRules.py:294` CoG defuzzification
- Vehicle-side forwarding
  - `ELITE-fusion/Node.py:237` `intra_area(...)`
  - `ELITE-fusion/Node.py:350` `find_next(...)`
  - `ELITE-fusion/GPSR.py:1` helpers for greedy/perimeter

Known Missing Assets (must be provided or replaced)
- `ELITE-fusion/traffic.py:32` `de_intersection.xml`
- `ELITE-fusion/traffic.py:47` `de_edges.xml`
- `ELITE-fusion/test.py:31,36,41,46,51,66` `de-*.mobility.tcl`

Summary
- This project contains the fusion and path-planning skeleton of ELITE and enough scaffolding to demonstrate junction-level routing with vehicle-side greedy forwarding. However, it does not fully reproduce the paper: the core parallel Q-learning (SPL) and adaptive policy deployment (APN) are missing, RSU/benchmark schemes are absent, and required simulation assets are not included. Filling these gaps (SPL/APN, assets, and evaluation) will bring the code much closer to a complete reproduction and enable apples-to-apples comparisons with the paper’s results.

