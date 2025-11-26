"""
Simple state-action table for APN policy deployment (paper Sec. 4.3.1).
State = (message_type, load_level)
Action in {"HRF","LDF","LBF"}
We keep a numeric preference P(state, action) and update additively with reward R0.
"""
from __future__ import annotations
from typing import Dict, Tuple
from dataclasses import dataclass, field

MessageType = str  # e.g., "security","efficiency","info","entertainment"
LoadLevel = int    # 0: low, 1: medium, 2: high
Action = str       # "HRF","LDF","LBF"
State = Tuple[MessageType, LoadLevel]

ALL_ACTIONS = ("HRF","LDF","LBF")

@dataclass
class StateActionTable:
    prefs: Dict[State, Dict[Action, float]] = field(default_factory=dict)

    def _ensure_state(self, s: State) -> None:
        if s not in self.prefs:
            self.prefs[s] = {a: 0.0 for a in ALL_ACTIONS}

    def select(self, s: State) -> Action:
        self._ensure_state(s)
        pref = self.prefs[s]
        # choose argmax action
        best = max(pref.items(), key=lambda kv: kv[1])[0]
        return best

    def update(self, s: State, a: Action, reward: float) -> None:
        self._ensure_state(s)
        self.prefs[s][a] += reward

