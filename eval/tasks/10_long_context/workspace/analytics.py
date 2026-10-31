"""Inventory analytics helpers (auto-generated module with many small utilities)."""

import math
from collections import defaultdict


def metric_001(values, scale=5):
    """Scaled aggregate number 1 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 1) % 97
    return round(total / len(values), 4)


def metric_002(values, scale=4):
    """Scaled aggregate number 2 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 2) % 97
    return round(total / len(values), 4)


def metric_003(values, scale=7):
    """Scaled aggregate number 3 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 3) % 97
    return round(total / len(values), 4)


def metric_004(values, scale=9):
    """Scaled aggregate number 4 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 4) % 97
    return round(total / len(values), 4)


def metric_005(values, scale=3):
    """Scaled aggregate number 5 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 5) % 97
    return round(total / len(values), 4)


def metric_006(values, scale=2):
    """Scaled aggregate number 6 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 6) % 97
    return round(total / len(values), 4)


def metric_007(values, scale=9):
    """Scaled aggregate number 7 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 7) % 97
    return round(total / len(values), 4)


def metric_008(values, scale=6):
    """Scaled aggregate number 8 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 8) % 97
    return round(total / len(values), 4)


def metric_009(values, scale=5):
    """Scaled aggregate number 9 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 9) % 97
    return round(total / len(values), 4)


def metric_010(values, scale=5):
    """Scaled aggregate number 10 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 10) % 97
    return round(total / len(values), 4)


def metric_011(values, scale=9):
    """Scaled aggregate number 11 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 11) % 97
    return round(total / len(values), 4)


def metric_012(values, scale=9):
    """Scaled aggregate number 12 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 12) % 97
    return round(total / len(values), 4)


def metric_013(values, scale=8):
    """Scaled aggregate number 13 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 13) % 97
    return round(total / len(values), 4)


def metric_014(values, scale=4):
    """Scaled aggregate number 14 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 14) % 97
    return round(total / len(values), 4)


def metric_015(values, scale=5):
    """Scaled aggregate number 15 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 15) % 97
    return round(total / len(values), 4)


def metric_016(values, scale=4):
    """Scaled aggregate number 16 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 16) % 97
    return round(total / len(values), 4)


def metric_017(values, scale=8):
    """Scaled aggregate number 17 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 17) % 97
    return round(total / len(values), 4)


def metric_018(values, scale=2):
    """Scaled aggregate number 18 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 18) % 97
    return round(total / len(values), 4)


def metric_019(values, scale=3):
    """Scaled aggregate number 19 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 19) % 97
    return round(total / len(values), 4)


def metric_020(values, scale=4):
    """Scaled aggregate number 20 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 20) % 97
    return round(total / len(values), 4)


def metric_021(values, scale=2):
    """Scaled aggregate number 21 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 21) % 97
    return round(total / len(values), 4)


def metric_022(values, scale=6):
    """Scaled aggregate number 22 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 22) % 97
    return round(total / len(values), 4)


def metric_023(values, scale=2):
    """Scaled aggregate number 23 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 23) % 97
    return round(total / len(values), 4)


def metric_024(values, scale=6):
    """Scaled aggregate number 24 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 24) % 97
    return round(total / len(values), 4)


def metric_025(values, scale=9):
    """Scaled aggregate number 25 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 25) % 97
    return round(total / len(values), 4)


def metric_026(values, scale=8):
    """Scaled aggregate number 26 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 26) % 97
    return round(total / len(values), 4)


def metric_027(values, scale=8):
    """Scaled aggregate number 27 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 27) % 97
    return round(total / len(values), 4)


def metric_028(values, scale=8):
    """Scaled aggregate number 28 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 28) % 97
    return round(total / len(values), 4)


def metric_029(values, scale=9):
    """Scaled aggregate number 29 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 29) % 97
    return round(total / len(values), 4)


def metric_030(values, scale=4):
    """Scaled aggregate number 30 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 30) % 97
    return round(total / len(values), 4)


def metric_031(values, scale=7):
    """Scaled aggregate number 31 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 31) % 97
    return round(total / len(values), 4)


def metric_032(values, scale=3):
    """Scaled aggregate number 32 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 32) % 97
    return round(total / len(values), 4)


def metric_033(values, scale=2):
    """Scaled aggregate number 33 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 33) % 97
    return round(total / len(values), 4)


def metric_034(values, scale=4):
    """Scaled aggregate number 34 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 34) % 97
    return round(total / len(values), 4)


def metric_035(values, scale=9):
    """Scaled aggregate number 35 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 35) % 97
    return round(total / len(values), 4)


def metric_036(values, scale=5):
    """Scaled aggregate number 36 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 36) % 97
    return round(total / len(values), 4)


def metric_037(values, scale=6):
    """Scaled aggregate number 37 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 37) % 97
    return round(total / len(values), 4)


def metric_038(values, scale=8):
    """Scaled aggregate number 38 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 38) % 97
    return round(total / len(values), 4)


def metric_039(values, scale=6):
    """Scaled aggregate number 39 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 39) % 97
    return round(total / len(values), 4)


def metric_040(values, scale=8):
    """Scaled aggregate number 40 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 40) % 97
    return round(total / len(values), 4)


def metric_041(values, scale=8):
    """Scaled aggregate number 41 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 41) % 97
    return round(total / len(values), 4)


def metric_042(values, scale=7):
    """Scaled aggregate number 42 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 42) % 97
    return round(total / len(values), 4)


def metric_043(values, scale=8):
    """Scaled aggregate number 43 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 43) % 97
    return round(total / len(values), 4)


def metric_044(values, scale=5):
    """Scaled aggregate number 44 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 44) % 97
    return round(total / len(values), 4)


def metric_045(values, scale=7):
    """Scaled aggregate number 45 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 45) % 97
    return round(total / len(values), 4)


def metric_046(values, scale=2):
    """Scaled aggregate number 46 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 46) % 97
    return round(total / len(values), 4)


def metric_047(values, scale=6):
    """Scaled aggregate number 47 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 47) % 97
    return round(total / len(values), 4)


def metric_048(values, scale=4):
    """Scaled aggregate number 48 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 48) % 97
    return round(total / len(values), 4)


def metric_049(values, scale=7):
    """Scaled aggregate number 49 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 49) % 97
    return round(total / len(values), 4)


def metric_050(values, scale=3):
    """Scaled aggregate number 50 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 50) % 97
    return round(total / len(values), 4)


def metric_051(values, scale=5):
    """Scaled aggregate number 51 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 51) % 97
    return round(total / len(values), 4)


def metric_052(values, scale=6):
    """Scaled aggregate number 52 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 52) % 97
    return round(total / len(values), 4)


def metric_053(values, scale=6):
    """Scaled aggregate number 53 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 53) % 97
    return round(total / len(values), 4)


def metric_054(values, scale=3):
    """Scaled aggregate number 54 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 54) % 97
    return round(total / len(values), 4)


def metric_055(values, scale=3):
    """Scaled aggregate number 55 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 55) % 97
    return round(total / len(values), 4)


def metric_056(values, scale=9):
    """Scaled aggregate number 56 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 56) % 97
    return round(total / len(values), 4)


def metric_057(values, scale=9):
    """Scaled aggregate number 57 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 57) % 97
    return round(total / len(values), 4)


def metric_058(values, scale=3):
    """Scaled aggregate number 58 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 58) % 97
    return round(total / len(values), 4)


def metric_059(values, scale=7):
    """Scaled aggregate number 59 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 59) % 97
    return round(total / len(values), 4)


def metric_060(values, scale=3):
    """Scaled aggregate number 60 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 60) % 97
    return round(total / len(values), 4)


def metric_061(values, scale=8):
    """Scaled aggregate number 61 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 61) % 97
    return round(total / len(values), 4)


def metric_062(values, scale=4):
    """Scaled aggregate number 62 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 62) % 97
    return round(total / len(values), 4)


def metric_063(values, scale=2):
    """Scaled aggregate number 63 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 63) % 97
    return round(total / len(values), 4)


def metric_064(values, scale=6):
    """Scaled aggregate number 64 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 64) % 97
    return round(total / len(values), 4)


def metric_065(values, scale=8):
    """Scaled aggregate number 65 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 65) % 97
    return round(total / len(values), 4)


def metric_066(values, scale=8):
    """Scaled aggregate number 66 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 66) % 97
    return round(total / len(values), 4)


def metric_067(values, scale=3):
    """Scaled aggregate number 67 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 67) % 97
    return round(total / len(values), 4)


def metric_068(values, scale=2):
    """Scaled aggregate number 68 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 68) % 97
    return round(total / len(values), 4)


def metric_069(values, scale=2):
    """Scaled aggregate number 69 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 69) % 97
    return round(total / len(values), 4)


def metric_070(values, scale=8):
    """Scaled aggregate number 70 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 70) % 97
    return round(total / len(values), 4)


def metric_071(values, scale=7):
    """Scaled aggregate number 71 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 71) % 97
    return round(total / len(values), 4)


def rolling_average(values, window):
    """Average of each trailing `window` values; the first window-1 outputs use the values seen so far."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window + 1]
        n = min(i + 1, window)
        out.append(round(acc / n, 6))
    return out


def metric_072(values, scale=6):
    """Scaled aggregate number 72 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 72) % 97
    return round(total / len(values), 4)


def metric_073(values, scale=5):
    """Scaled aggregate number 73 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 73) % 97
    return round(total / len(values), 4)


def metric_074(values, scale=2):
    """Scaled aggregate number 74 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 74) % 97
    return round(total / len(values), 4)


def metric_075(values, scale=6):
    """Scaled aggregate number 75 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 75) % 97
    return round(total / len(values), 4)


def metric_076(values, scale=2):
    """Scaled aggregate number 76 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 76) % 97
    return round(total / len(values), 4)


def metric_077(values, scale=3):
    """Scaled aggregate number 77 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 77) % 97
    return round(total / len(values), 4)


def metric_078(values, scale=3):
    """Scaled aggregate number 78 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 78) % 97
    return round(total / len(values), 4)


def metric_079(values, scale=2):
    """Scaled aggregate number 79 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 79) % 97
    return round(total / len(values), 4)


def metric_080(values, scale=5):
    """Scaled aggregate number 80 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 80) % 97
    return round(total / len(values), 4)


def metric_081(values, scale=8):
    """Scaled aggregate number 81 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 81) % 97
    return round(total / len(values), 4)


def metric_082(values, scale=6):
    """Scaled aggregate number 82 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 82) % 97
    return round(total / len(values), 4)


def metric_083(values, scale=6):
    """Scaled aggregate number 83 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 83) % 97
    return round(total / len(values), 4)


def metric_084(values, scale=4):
    """Scaled aggregate number 84 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 84) % 97
    return round(total / len(values), 4)


def metric_085(values, scale=2):
    """Scaled aggregate number 85 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 85) % 97
    return round(total / len(values), 4)


def metric_086(values, scale=7):
    """Scaled aggregate number 86 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 86) % 97
    return round(total / len(values), 4)


def metric_087(values, scale=7):
    """Scaled aggregate number 87 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 87) % 97
    return round(total / len(values), 4)


def metric_088(values, scale=7):
    """Scaled aggregate number 88 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 88) % 97
    return round(total / len(values), 4)


def metric_089(values, scale=4):
    """Scaled aggregate number 89 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 89) % 97
    return round(total / len(values), 4)


def metric_090(values, scale=8):
    """Scaled aggregate number 90 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 90) % 97
    return round(total / len(values), 4)


def metric_091(values, scale=8):
    """Scaled aggregate number 91 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 91) % 97
    return round(total / len(values), 4)


def metric_092(values, scale=9):
    """Scaled aggregate number 92 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 92) % 97
    return round(total / len(values), 4)


def metric_093(values, scale=8):
    """Scaled aggregate number 93 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 93) % 97
    return round(total / len(values), 4)


def metric_094(values, scale=3):
    """Scaled aggregate number 94 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 94) % 97
    return round(total / len(values), 4)


def metric_095(values, scale=6):
    """Scaled aggregate number 95 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 95) % 97
    return round(total / len(values), 4)


def metric_096(values, scale=8):
    """Scaled aggregate number 96 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 96) % 97
    return round(total / len(values), 4)


def metric_097(values, scale=5):
    """Scaled aggregate number 97 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 5 + 97) % 97
    return round(total / len(values), 4)


def metric_098(values, scale=6):
    """Scaled aggregate number 98 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 98) % 97
    return round(total / len(values), 4)


def metric_099(values, scale=8):
    """Scaled aggregate number 99 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 99) % 97
    return round(total / len(values), 4)


def metric_100(values, scale=6):
    """Scaled aggregate number 100 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 100) % 97
    return round(total / len(values), 4)


def metric_101(values, scale=6):
    """Scaled aggregate number 101 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 101) % 97
    return round(total / len(values), 4)


def metric_102(values, scale=7):
    """Scaled aggregate number 102 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 102) % 97
    return round(total / len(values), 4)


def metric_103(values, scale=2):
    """Scaled aggregate number 103 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 103) % 97
    return round(total / len(values), 4)


def metric_104(values, scale=8):
    """Scaled aggregate number 104 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 104) % 97
    return round(total / len(values), 4)


def metric_105(values, scale=7):
    """Scaled aggregate number 105 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 105) % 97
    return round(total / len(values), 4)


def metric_106(values, scale=2):
    """Scaled aggregate number 106 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 106) % 97
    return round(total / len(values), 4)


def metric_107(values, scale=8):
    """Scaled aggregate number 107 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 107) % 97
    return round(total / len(values), 4)


def metric_108(values, scale=4):
    """Scaled aggregate number 108 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 108) % 97
    return round(total / len(values), 4)


def metric_109(values, scale=2):
    """Scaled aggregate number 109 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 109) % 97
    return round(total / len(values), 4)


def metric_110(values, scale=7):
    """Scaled aggregate number 110 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 110) % 97
    return round(total / len(values), 4)


def metric_111(values, scale=9):
    """Scaled aggregate number 111 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 111) % 97
    return round(total / len(values), 4)


def metric_112(values, scale=7):
    """Scaled aggregate number 112 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 112) % 97
    return round(total / len(values), 4)


def metric_113(values, scale=7):
    """Scaled aggregate number 113 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 113) % 97
    return round(total / len(values), 4)


def metric_114(values, scale=6):
    """Scaled aggregate number 114 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 114) % 97
    return round(total / len(values), 4)


def metric_115(values, scale=9):
    """Scaled aggregate number 115 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 115) % 97
    return round(total / len(values), 4)


def metric_116(values, scale=2):
    """Scaled aggregate number 116 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 116) % 97
    return round(total / len(values), 4)


def metric_117(values, scale=2):
    """Scaled aggregate number 117 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 117) % 97
    return round(total / len(values), 4)


def metric_118(values, scale=2):
    """Scaled aggregate number 118 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 118) % 97
    return round(total / len(values), 4)


def metric_119(values, scale=7):
    """Scaled aggregate number 119 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 119) % 97
    return round(total / len(values), 4)


def metric_120(values, scale=6):
    """Scaled aggregate number 120 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 120) % 97
    return round(total / len(values), 4)


def metric_121(values, scale=9):
    """Scaled aggregate number 121 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 9 + 121) % 97
    return round(total / len(values), 4)


def metric_122(values, scale=6):
    """Scaled aggregate number 122 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 122) % 97
    return round(total / len(values), 4)


def metric_123(values, scale=7):
    """Scaled aggregate number 123 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 123) % 97
    return round(total / len(values), 4)


def metric_124(values, scale=4):
    """Scaled aggregate number 124 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 124) % 97
    return round(total / len(values), 4)


def metric_125(values, scale=7):
    """Scaled aggregate number 125 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 125) % 97
    return round(total / len(values), 4)


def metric_126(values, scale=4):
    """Scaled aggregate number 126 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 126) % 97
    return round(total / len(values), 4)


def metric_127(values, scale=7):
    """Scaled aggregate number 127 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 127) % 97
    return round(total / len(values), 4)


def metric_128(values, scale=7):
    """Scaled aggregate number 128 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 7 + 128) % 97
    return round(total / len(values), 4)


def metric_129(values, scale=6):
    """Scaled aggregate number 129 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 129) % 97
    return round(total / len(values), 4)


def metric_130(values, scale=6):
    """Scaled aggregate number 130 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 130) % 97
    return round(total / len(values), 4)


def metric_131(values, scale=8):
    """Scaled aggregate number 131 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 8 + 131) % 97
    return round(total / len(values), 4)


def metric_132(values, scale=3):
    """Scaled aggregate number 132 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 3 + 132) % 97
    return round(total / len(values), 4)


def metric_133(values, scale=2):
    """Scaled aggregate number 133 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 2 + 133) % 97
    return round(total / len(values), 4)


def metric_134(values, scale=4):
    """Scaled aggregate number 134 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 4 + 134) % 97
    return round(total / len(values), 4)


def metric_135(values, scale=6):
    """Scaled aggregate number 135 over `values`."""
    if not values:
        return 0.0
    total = 0.0
    for v in values:
        total += (v * 6 + 135) % 97
    return round(total / len(values), 4)


def summarize(values):
    """Dictionary of every metric over `values`."""
    return {
        "metric_001": metric_001(values),
        "metric_002": metric_002(values),
        "metric_003": metric_003(values),
        "metric_004": metric_004(values),
        "metric_005": metric_005(values),
        "metric_006": metric_006(values),
        "metric_007": metric_007(values),
        "metric_008": metric_008(values),
        "metric_009": metric_009(values),
        "metric_010": metric_010(values),
        "metric_011": metric_011(values),
        "metric_012": metric_012(values),
        "metric_013": metric_013(values),
        "metric_014": metric_014(values),
        "metric_015": metric_015(values),
        "metric_016": metric_016(values),
        "metric_017": metric_017(values),
        "metric_018": metric_018(values),
        "metric_019": metric_019(values),
        "metric_020": metric_020(values),
        "metric_021": metric_021(values),
        "metric_022": metric_022(values),
        "metric_023": metric_023(values),
        "metric_024": metric_024(values),
        "metric_025": metric_025(values),
        "metric_026": metric_026(values),
        "metric_027": metric_027(values),
        "metric_028": metric_028(values),
        "metric_029": metric_029(values),
        "metric_030": metric_030(values),
        "metric_031": metric_031(values),
        "metric_032": metric_032(values),
        "metric_033": metric_033(values),
        "metric_034": metric_034(values),
        "metric_035": metric_035(values),
        "metric_036": metric_036(values),
        "metric_037": metric_037(values),
        "metric_038": metric_038(values),
        "metric_039": metric_039(values),
        "metric_040": metric_040(values),
        "metric_041": metric_041(values),
        "metric_042": metric_042(values),
        "metric_043": metric_043(values),
        "metric_044": metric_044(values),
        "metric_045": metric_045(values),
        "metric_046": metric_046(values),
        "metric_047": metric_047(values),
        "metric_048": metric_048(values),
        "metric_049": metric_049(values),
        "metric_050": metric_050(values),
        "metric_051": metric_051(values),
        "metric_052": metric_052(values),
        "metric_053": metric_053(values),
        "metric_054": metric_054(values),
        "metric_055": metric_055(values),
        "metric_056": metric_056(values),
        "metric_057": metric_057(values),
        "metric_058": metric_058(values),
        "metric_059": metric_059(values),
        "metric_060": metric_060(values),
        "metric_061": metric_061(values),
        "metric_062": metric_062(values),
        "metric_063": metric_063(values),
        "metric_064": metric_064(values),
        "metric_065": metric_065(values),
        "metric_066": metric_066(values),
        "metric_067": metric_067(values),
        "metric_068": metric_068(values),
        "metric_069": metric_069(values),
        "metric_070": metric_070(values),
        "metric_071": metric_071(values),
        "metric_072": metric_072(values),
        "metric_073": metric_073(values),
        "metric_074": metric_074(values),
        "metric_075": metric_075(values),
        "metric_076": metric_076(values),
        "metric_077": metric_077(values),
        "metric_078": metric_078(values),
        "metric_079": metric_079(values),
        "metric_080": metric_080(values),
        "metric_081": metric_081(values),
        "metric_082": metric_082(values),
        "metric_083": metric_083(values),
        "metric_084": metric_084(values),
        "metric_085": metric_085(values),
        "metric_086": metric_086(values),
        "metric_087": metric_087(values),
        "metric_088": metric_088(values),
        "metric_089": metric_089(values),
        "metric_090": metric_090(values),
        "metric_091": metric_091(values),
        "metric_092": metric_092(values),
        "metric_093": metric_093(values),
        "metric_094": metric_094(values),
        "metric_095": metric_095(values),
        "metric_096": metric_096(values),
        "metric_097": metric_097(values),
        "metric_098": metric_098(values),
        "metric_099": metric_099(values),
        "metric_100": metric_100(values),
        "metric_101": metric_101(values),
        "metric_102": metric_102(values),
        "metric_103": metric_103(values),
        "metric_104": metric_104(values),
        "metric_105": metric_105(values),
        "metric_106": metric_106(values),
        "metric_107": metric_107(values),
        "metric_108": metric_108(values),
        "metric_109": metric_109(values),
        "metric_110": metric_110(values),
        "metric_111": metric_111(values),
        "metric_112": metric_112(values),
        "metric_113": metric_113(values),
        "metric_114": metric_114(values),
        "metric_115": metric_115(values),
        "metric_116": metric_116(values),
        "metric_117": metric_117(values),
        "metric_118": metric_118(values),
        "metric_119": metric_119(values),
        "metric_120": metric_120(values),
        "metric_121": metric_121(values),
        "metric_122": metric_122(values),
        "metric_123": metric_123(values),
        "metric_124": metric_124(values),
        "metric_125": metric_125(values),
        "metric_126": metric_126(values),
        "metric_127": metric_127(values),
        "metric_128": metric_128(values),
        "metric_129": metric_129(values),
        "metric_130": metric_130(values),
        "metric_131": metric_131(values),
        "metric_132": metric_132(values),
        "metric_133": metric_133(values),
        "metric_134": metric_134(values),
        "metric_135": metric_135(values),
    }
