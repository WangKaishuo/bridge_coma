"""Contract-bridge score calculation."""

from dataclasses import dataclass


@dataclass
class Contract:
    """Final auction contract."""

    level: int
    suit: int
    doubled: int
    declarer: int

    @property
    def required_tricks(self) -> int:
        return 6 + self.level

    def __str__(self) -> str:
        strains = ["C", "D", "H", "S", "NT"]
        players = ["N", "E", "S", "W"]
        suffix = ["", "X", "XX"][self.doubled]
        return f"{self.level}{strains[self.suit]}{suffix} by {players[self.declarer]}"


def calculate_score(contract: Contract, tricks: int, vulnerable: bool) -> int:
    """Calculate duplicate score from the declarer's perspective."""
    result = tricks - contract.required_tricks
    return (
        _score_made(contract, result, vulnerable)
        if result >= 0
        else _score_down(contract, -result, vulnerable)
    )


def _score_made(contract: Contract, overtricks: int, vulnerable: bool) -> int:
    level, suit, doubled = contract.level, contract.suit, contract.doubled
    trick_value = 20 if suit <= 1 else 30
    contract_points = 40 + (level - 1) * 30 if suit == 4 else level * trick_value
    contract_points *= (1, 2, 4)[doubled]
    bonus = (500 if vulnerable else 300) if contract_points >= 100 else 50
    if doubled == 1:
        bonus += 50
    elif doubled == 2:
        bonus += 100
    if level == 6:
        bonus += 750 if vulnerable else 500
    elif level == 7:
        bonus += 1500 if vulnerable else 1000
    if doubled == 0:
        overtrick_value = trick_value
    elif doubled == 1:
        overtrick_value = 200 if vulnerable else 100
    else:
        overtrick_value = 400 if vulnerable else 200
    return contract_points + bonus + overtricks * overtrick_value


def _score_down(contract: Contract, undertricks: int, vulnerable: bool) -> int:
    if contract.doubled == 0:
        return -undertricks * (100 if vulnerable else 50)
    penalties = [200] + [300] * 12 if vulnerable else [100, 200, 200] + [300] * 10
    total = sum(penalties[:undertricks])
    return -total * (2 if contract.doubled == 2 else 1)
