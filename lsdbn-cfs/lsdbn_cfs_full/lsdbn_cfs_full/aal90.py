from __future__ import annotations

from typing import Dict, List, Tuple

AAL90_NAMES: List[str] = [
    "PreCG.L", "PreCG.R", "SFGdor.L", "SFGdor.R", "ORBsup.L", "ORBsup.R",
    "MFG.L", "MFG.R", "ORBmid.L", "ORBmid.R", "IFGoperc.L", "IFGoperc.R",
    "IFGtriang.L", "IFGtriang.R", "ORBinf.L", "ORBinf.R", "ROL.L", "ROL.R",
    "SMA.L", "SMA.R", "OLF.L", "OLF.R", "SFGmed.L", "SFGmed.R",
    "ORBsupmed.L", "ORBsupmed.R", "REC.L", "REC.R", "INS.L", "INS.R",
    "ACG.L", "ACG.R", "DCG.L", "DCG.R", "PCG.L", "PCG.R",
    "HIP.L", "HIP.R", "PHG.L", "PHG.R", "AMYG.L", "AMYG.R",
    "CAL.L", "CAL.R", "CUN.L", "CUN.R", "LING.L", "LING.R",
    "SOG.L", "SOG.R", "MOG.L", "MOG.R", "IOG.L", "IOG.R",
    "FFG.L", "FFG.R", "PoCG.L", "PoCG.R", "SPG.L", "SPG.R",
    "IPL.L", "IPL.R", "SMG.L", "SMG.R", "ANG.L", "ANG.R",
    "PCUN.L", "PCUN.R", "PCL.L", "PCL.R", "CAU.L", "CAU.R",
    "PUT.L", "PUT.R", "PAL.L", "PAL.R", "THA.L", "THA.R",
    "HES.L", "HES.R", "STG.L", "STG.R", "TPOsup.L", "TPOsup.R",
    "MTG.L", "MTG.R", "TPOmid.L", "TPOmid.R", "ITG.L", "ITG.R",
]


def _network_for_roi(name: str) -> str:
    base = name.split(".")[0]
    if base in {"CAL", "CUN", "LING", "SOG", "MOG", "IOG", "FFG"}:
        return "VIS"
    if base in {"PreCG", "PoCG", "SMA", "PCL", "ROL"}:
        return "SMN"
    if base in {"SPG", "IPL", "SMG", "ANG"}:
        return "DAN"
    if base in {"ACG", "DCG", "INS", "AMYG"}:
        return "VAN"
    if base in {"HIP", "PHG", "ORBsup", "ORBmid", "ORBinf", "ORBsupmed", "REC", "OLF"}:
        return "LIM"
    if base in {"SFGdor", "MFG", "IFGoperc", "IFGtriang", "SFGmed"}:
        return "FPN"
    if base in {"CAU", "PUT", "PAL", "THA"}:
        return "SUB"
    return "DMN"


ROI_NETWORK: Dict[str, str] = {name: _network_for_roi(name) for name in AAL90_NAMES}
NETWORK_ORDER = ["VIS", "SMN", "DAN", "VAN", "LIM", "FPN", "DMN", "SUB"]


def feature_index_to_edge(feature_id: int, n_roi: int = 90) -> Tuple[int, int]:
    """Map a 1-based no-self feature ID to a 1-based directed ROI edge."""
    if feature_id < 1 or feature_id > n_roi * (n_roi - 1):
        raise ValueError("feature_id must be in [1, %d]" % (n_roi * (n_roi - 1)))
    zero = feature_id - 1
    src = zero // (n_roi - 1) + 1
    rem = zero % (n_roi - 1) + 1
    tgt = rem if rem < src else rem + 1
    return src, tgt


def edge_to_feature_index(src: int, tgt: int, n_roi: int = 90) -> int:
    """Map a 1-based directed ROI edge to a 1-based no-self feature ID."""
    if src == tgt:
        raise ValueError("Self-connections are excluded.")
    if not (1 <= src <= n_roi and 1 <= tgt <= n_roi):
        raise ValueError("ROI indices must be 1-based and within [1, %d]." % n_roi)
    return (src - 1) * (n_roi - 1) + (tgt if tgt < src else tgt - 1)


def edge_name(src: int, tgt: int) -> str:
    return "%s->%s" % (AAL90_NAMES[src - 1], AAL90_NAMES[tgt - 1])

