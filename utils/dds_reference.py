"""Audited double-dummy par reference and actor-oriented task utility.

The project stores DDS tricks in ``(C, D, H, S, NT) x (N, E, S, W)``
order.  Endplay's DDS binding uses ``(S, H, D, C, NT)`` strain order.  This
module is the single conversion boundary and returns the dealer-par score from
North-South's perspective.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from utils.imp import score_to_imp


DDS_PAR_REFERENCE_KIND = "dds_dealer_par_v1"
ACTING_PARTNERSHIP_UTILITY = "acting_partnership_imp_v1"


def partnership_sign(seat: int) -> int:
    """Return +1 for N/S and -1 for E/W."""
    seat = int(seat)
    if not 0 <= seat < 4:
        raise ValueError("seat must lie in [0, 4)")
    return 1 if seat % 2 == 0 else -1


def actor_duplicate_imp(
    terminal_score_ns: int, reference_score_ns: int, acting_seat: int
) -> int:
    """Convert an NS score difference to the acting partnership's IMP utility."""
    imp_ns = int(score_to_imp(int(terminal_score_ns) - int(reference_score_ns)))
    return partnership_sign(acting_seat) * imp_ns


def _endplay_vulnerability(vulnerability: Sequence[bool]):
    from endplay.types import Vul

    if len(vulnerability) != 2:
        raise ValueError("vulnerability must contain (NS, EW)")
    ns, ew = (bool(value) for value in vulnerability)
    if ns and ew:
        return Vul.both
    if ns:
        return Vul.ns
    if ew:
        return Vul.ew
    return Vul.none


def dds_par_score_ns(
    dd_table: np.ndarray, vulnerability: Sequence[bool], dealer: int
) -> int:
    """Return the DDS dealer-par score from North-South's perspective.

    Dealer and vulnerability are part of the reference identity.  The result
    is zero-sum: swapping N/S with E/W, rotating the dealer by one seat, and
    swapping vulnerability negates the score.
    """
    table = np.asarray(dd_table)
    if table.shape != (5, 4):
        raise ValueError("dd_table must have shape (5, 4)")
    if not np.issubdtype(table.dtype, np.number) or not np.all(np.isfinite(table)):
        raise ValueError("dd_table must contain finite numeric trick counts")
    if np.any(table < 0) or np.any(table > 13) or not np.all(table == np.floor(table)):
        raise ValueError("dd_table trick counts must be integers in [0, 13]")
    dealer = int(dealer)
    if not 0 <= dealer < 4:
        raise ValueError("dealer must lie in [0, 4)")

    # DDTable currently has no public constructor from raw results, so create
    # the binding's value object explicitly and keep that private dependency
    # isolated here.
    import endplay._dds as _dds
    from endplay.dds import par
    from endplay.dds.ddtable import DDTable
    from endplay.types import Player

    raw = _dds.ddTableResults()
    # project C,D,H,S,NT -> DDS S,H,D,C,NT
    dds_strain_by_project_strain = (3, 2, 1, 0, 4)
    for project_strain, dds_strain in enumerate(dds_strain_by_project_strain):
        for player in range(4):
            raw.resTable[dds_strain][player] = int(table[project_strain, player])
    result = par(
        DDTable(raw),
        _endplay_vulnerability(vulnerability),
        Player(dealer),
    )
    return int(result.score)
