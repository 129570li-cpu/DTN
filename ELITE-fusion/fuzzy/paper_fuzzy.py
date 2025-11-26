"""
Fuzzy fusion strictly following the paper's description:
- Inputs (normalized to (0,1]): PRR (R), AD (D), HC (H), RC (S)
  * R sets: {Poor, Medium, Good}
  * D sets: {Short, Middle, Long}
  * H sets: {Bad, Medium, Good}  (Good means fewer hops is good)
  * S sets: {Low, Medium, High}
- Output Grade: {Worst, Bad, Medium, Good, Excellent}
- Membership: triangular; intersections at (0.2*mv, 0.5*mv, 0.8*mv) on x-axis; since inputs are normalized via Qnorm in (0,1], we set mv=1.
- Inference: Mamdani Min-Max
- Rule generation: numeric approach in the paper:
  For a policy, choose 3 inputs (e.g. HRF uses R, D, S) and assign weights W={1,1,1};
  Each antecedent's level maps to score S ∈ {1/3, 2/3, 1} (level1, level2, level3).
  g = W1*S1 + W2*S2 + W3*S3 in [0,1], then consequent Grade is:
    [0,0.2) -> Worst; [0.2,0.4) -> Bad; [0.4,0.6) -> Medium; [0.6,0.8) -> Good; [0.8,1] -> Excellent
We construct all 3x3x3 rules by levels and use the Min of membership degrees of antecedents as rule strength; aggregate by Max per Grade; defuzzify by Center of Gravity on Grade's triangular sets over [0,1].
"""
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np

# Triangular membership helper
def tri(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

# Input membership sets over [0,1]
def prr_sets(x: float) -> Dict[str, float]:
    # Poor (center near 0.2), Medium (0.5), Good (0.8)
    return {
        "Poor":   tri(x, 0.0, 0.2, 0.5),
        "Medium": tri(x, 0.2, 0.5, 0.8),
        "Good":   tri(x, 0.5, 0.8, 1.0),
    }

def ad_sets(x: float) -> Dict[str, float]:
    # Short near 0.2, Middle 0.5, Long 0.8 (smaller is better)
    return {
        "Short":  tri(x, 0.0, 0.2, 0.5),
        "Middle": tri(x, 0.2, 0.5, 0.8),
        "Long":   tri(x, 0.5, 0.8, 1.0),
    }

def hc_sets(x: float) -> Dict[str, float]:
    # Good (few hops) near 0.2, Medium 0.5, Bad 0.8
    return {
        "Good":   tri(x, 0.0, 0.2, 0.5),
        "Medium": tri(x, 0.2, 0.5, 0.8),
        "Bad":    tri(x, 0.5, 0.8, 1.0),
    }

def rc_sets(x: float) -> Dict[str, float]:
    # Low near 0.2, Medium 0.5, High 0.8
    return {
        "Low":    tri(x, 0.0, 0.2, 0.5),
        "Medium": tri(x, 0.2, 0.5, 0.8),
        "High":   tri(x, 0.5, 0.8, 1.0),
    }

# Grade sets over [0,1]
def grade_sets(x: float) -> Dict[str, float]:
    return {
        "Worst":     tri(x, 0.0, 0.1, 0.2),
        "Bad":       tri(x, 0.2, 0.3, 0.4),
        "Medium":    tri(x, 0.4, 0.5, 0.6),
        "Good":      tri(x, 0.6, 0.7, 0.8),
        "Excellent": tri(x, 0.8, 0.9, 1.0),
    }

def _level_score(var: str, level_name: str) -> float:
    # Map membership set name to level 1/2/3 → score {1/3, 2/3, 1}
    # Following the paper's treatment for each variable
    if var == "R":  # PRR: Poor (1), Medium (2), Good (3)
        order = ["Poor", "Medium", "Good"]
    elif var == "D":  # AD: Short (3), Middle (2), Long (1) since smaller is better
        order = ["Long", "Middle", "Short"]
    elif var == "H":  # HC: Bad (1), Medium (2), Good (3) since fewer hops better
        order = ["Bad", "Medium", "Good"]
    elif var == "S":  # RC: Low (3), Medium (2), High (1)
        order = ["High", "Medium", "Low"]
    else:
        order = []
    try:
        idx = order.index(level_name)
    except ValueError:
        return 0.0
    level = idx + 1  # 1..3
    return {1: 1.0/3.0, 2: 2.0/3.0, 3: 1.0}[level]

def _grade_from_g(g: float) -> str:
    if 0.0 <= g < 0.2:
        return "Worst"
    if 0.2 <= g < 0.4:
        return "Bad"
    if 0.4 <= g < 0.6:
        return "Medium"
    if 0.6 <= g < 0.8:
        return "Good"
    return "Excellent"

def _defuzzy(minmax: Dict[str, float], step: float = 0.01) -> float:
    # Center of Gravity over [0,1]
    xs = np.arange(0.0, 1.0 + step, step)
    num = 0.0
    den = 0.0
    for x in xs:
        gs = grade_sets(x)
        mu = 0.0
        for gname, gmem in gs.items():
            mu = max(mu, min(minmax.get(gname, 0.0), gmem))
        num += x * mu
        den += mu
    return 0.0 if den == 0.0 else float(num / den)

def fuse_hrf(v_prr: float, v_ad: float, v_rc: float) -> float:
    # Inputs normalized in [0,1]; Build rule outputs via numeric approach
    R = prr_sets(v_prr)
    D = ad_sets(v_ad)
    S = rc_sets(v_rc)
    minmax: Dict[str, float] = {}
    for r_name, r_mem in R.items():
        for d_name, d_mem in D.items():
            for s_name, s_mem in S.items():
                strength = min(r_mem, d_mem, s_mem)
                if strength == 0.0:
                    continue
                g = ( _level_score("R", r_name) + _level_score("D", d_name) + _level_score("S", s_name) ) / 3.0
                grade = _grade_from_g(g)
                minmax[grade] = max(minmax.get(grade, 0.0), strength)
    return _defuzzy(minmax)

def fuse_ldf(v_ad: float, v_hc: float, v_prr: float) -> float:
    D = ad_sets(v_ad)
    H = hc_sets(v_hc)
    R = prr_sets(v_prr)
    minmax: Dict[str, float] = {}
    for d_name, d_mem in D.items():
        for h_name, h_mem in H.items():
            for r_name, r_mem in R.items():
                strength = min(d_mem, h_mem, r_mem)
                if strength == 0.0:
                    continue
                g = ( _level_score("D", d_name) + _level_score("H", h_name) + _level_score("R", r_name) ) / 3.0
                grade = _grade_from_g(g)
                minmax[grade] = max(minmax.get(grade, 0.0), strength)
    return _defuzzy(minmax)

def fuse_lbf(v_rc: float, v_hc: float, v_ad: float) -> float:
    S = rc_sets(v_rc)
    H = hc_sets(v_hc)
    D = ad_sets(v_ad)
    minmax: Dict[str, float] = {}
    for s_name, s_mem in S.items():
        for h_name, h_mem in H.items():
            for d_name, d_mem in D.items():
                strength = min(s_mem, h_mem, d_mem)
                if strength == 0.0:
                    continue
                g = ( _level_score("S", s_name) + _level_score("H", h_name) + _level_score("D", d_name) ) / 3.0
                grade = _grade_from_g(g)
                minmax[grade] = max(minmax.get(grade, 0.0), strength)
    return _defuzzy(minmax)

