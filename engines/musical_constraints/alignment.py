"""Syllable-to-note alignment by dynamic programming.

Given the sung note slots of the original line and the syllables of a candidate
adaptation, find the cheapest way to sing the candidate on the melody:

  1:1        one syllable on one note (ideal)
  melisma    one syllable stretched over 2-3 notes (acceptable, cheaper on long open vowels)
  crowding   two syllables squeezed on one note (costly, worse on short notes)
  elision    an optional syllable (French final schwa) is not sung (almost free)

The cost is explainable: every operation is recorded with its reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.models import MelodySlot, Syllable

MIN_SINGABLE = 0.09          # seconds; below this a syllable is hard to articulate
COST_MELISMA = 0.35
COST_CROWD = 0.9
COST_ELIDE = 0.05
COST_STRESS_WEAK = 0.4       # stressed syllable on a weak beat
COST_SHORT = 0.6


@dataclass
class Alignment:
    cost: float
    mapping: list[list[int]]                 # for each syllable, the slot indexes it uses
    operations: list[str] = field(default_factory=list)
    feasible: bool = True

    def durations(self, slots: list[MelodySlot], n_share: dict[int, int]) -> list[float]:
        out = []
        for idx in self.mapping:
            out.append(sum(slots[j].duration / n_share.get(j, 1) for j in idx))
        return out


def _stress_cost(syl: Syllable, slot: MelodySlot) -> float:
    if syl.stressed and slot.beat_strength < 0.5:
        return COST_STRESS_WEAK
    if not syl.stressed and slot.beat_strength >= 1.0:
        return 0.1
    return 0.0


def align(syllables: list[Syllable], slots: list[MelodySlot]) -> Alignment:
    n, m = len(syllables), len(slots)
    if m == 0:
        return Alignment(float(n), [[] for _ in syllables], ["aucune note de reference"], n == 0)
    INF = float("inf")
    # dp[i][j]: best cost having placed i syllables on the first j slots
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == INF:
                continue
            if i < n and syllables[i].optional:                        # elision
                if cur + COST_ELIDE < dp[i + 1][j]:
                    dp[i + 1][j] = cur + COST_ELIDE
                    back[i + 1][j] = (i, j, "elide")
            if i < n and j < m:                                        # 1:1 and melisma
                syl = syllables[i]
                for k in (1, 2, 3):
                    if j + k > m:
                        break
                    used = slots[j:j + k]
                    c = _stress_cost(syl, used[0])
                    if k > 1:
                        long_open = syl.open and sum(s.duration for s in used) > 0.35
                        c += COST_MELISMA * (k - 1) * (0.5 if long_open else 1.0)
                    if cur + c < dp[i + 1][j + k]:
                        dp[i + 1][j + k] = cur + c
                        back[i + 1][j + k] = (i, j, f"melisma{k}" if k > 1 else "one")
            if i + 1 < n and j < m:                                    # crowding: 2 syllables / 1 note
                slot = slots[j]
                c = COST_CROWD + (COST_SHORT if slot.duration / 2 < MIN_SINGABLE else 0.0)
                c += _stress_cost(syllables[i], slot)
                if cur + c < dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = cur + c
                    back[i + 2][j + 1] = (i, j, "crowd")
    if dp[n][m] == INF:
        return Alignment(INF, [[] for _ in syllables], ["alignement impossible"], False)

    mapping: list[list[int]] = [[] for _ in syllables]
    ops: list[str] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = back[i][j]
        if op == "elide":
            ops.append(f"'{syllables[pi].text}' non chantee (elision)")
        elif op == "crowd":
            mapping[pi] = [pj]
            mapping[pi + 1] = [pj]
            ops.append(f"'{syllables[pi].text}{syllables[pi + 1].text}' serrees sur une seule note")
        else:
            mapping[pi] = list(range(pj, j))
            if op.startswith("melisma"):
                ops.append(f"'{syllables[pi].text}' etiree sur {j - pj} notes")
        i, j = pi, pj
    ops.reverse()
    return Alignment(dp[n][m], mapping, ops, True)
