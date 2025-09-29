
"""
LAFA FAH preprocessing utilities

- Loads FAH sheets for 1994–1998, 2003–2004, 2005–2006, 2007–2008
- Detects Men/Women "Mean" columns and the data start row programmatically
- Extracts mean values for specified fruit and dairy categories
- Produces a tidy DataFrame: [period, product, gender, mean, group]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


PERIOD_LABELS = ["1994-1998", "2003-2004", "2005-2006", "2007-2008"]

FRUITS = [
    "Apples as fruit",
    "Bananas",
    "Berries",
    "Grapes",
    "Melons",
    "Oranges, Total",
    "Other citrus fruit",
    "Stone fruit",
    "Tropical fruit",
]

DAIRY = [
    "Fluid milk total",
    "Butter",
    "Cheese",
    "Yogurt",
    "Dairy, Other",
]

FAH_PAT = re.compile(r"FAH|at\s*home", re.IGNORECASE)
PERIOD_PATS = {
    "1994-1998": re.compile(r"94\s*[-–]\s*98", re.IGNORECASE),
    "2003-2004": re.compile(r"03\s*[-–]\s*04", re.IGNORECASE),
    "2005-2006": re.compile(r"05\s*[-–]\s*06", re.IGNORECASE),
    "2007-2008": re.compile(r"07\s*[-–]\s*08", re.IGNORECASE),
}


@dataclass
class GenderColInfo:
    header_row: int
    men_col: int
    women_col: int


def detect_fah_sheets(xlsx_path: str | Path, engine: str = "openpyxl") -> Dict[str, str]:
    """
    Return a mapping: period_label -> FAH sheet name
    """
    xls = pd.ExcelFile(xlsx_path, engine=engine)
    mapping: Dict[str, str] = {}
    for s in xls.sheet_names:
        if FAH_PAT.search(s):
            for period, pat in PERIOD_PATS.items():
                if pat.search(s):
                    mapping[period] = s
    # Ensure all periods exist
    missing = [p for p in PERIOD_LABELS if p not in mapping]
    if missing:
        raise ValueError(f"Could not detect FAH sheets for periods: {', '.join(missing)}. Found: {mapping}")
    return mapping


def detect_gender_cols(df: pd.DataFrame) -> GenderColInfo:
    """
    Detect header row & Men/Women mean columns using the 'Age and gender' block.
    Fallback to typical positions if not found.
    """
    men_idx: Optional[int] = None
    women_idx: Optional[int] = None
    header_row: Optional[int] = None

    max_scan = min(160, len(df) - 3)
    for r in range(max_scan):
        row = df.iloc[r].astype(str).tolist()
        if any("Age and gender" in str(x) for x in row):
            labels_row = df.iloc[r + 1].astype(str).tolist()  # Boys, Girls, Men, Women
            means_row = df.iloc[r + 2].astype(str).tolist()   # Mean, 95% CI
            # Candidate positions
            men_cands = [i for i, v in enumerate(labels_row) if str(v).strip().lower() == "men"]
            wom_cands = [i for i, v in enumerate(labels_row) if str(v).strip().lower() == "women"]
            if men_cands and str(means_row[men_cands[0]]).strip().lower() == "mean":
                men_idx = men_cands[0]
            if wom_cands and str(means_row[wom_cands[0]]).strip().lower() == "mean":
                women_idx = wom_cands[0]
            header_row = r + 3  # row of '% lower upper' -> data starts below
            break

    # Fallbacks derived from visual inspection hints
    if men_idx is None:
        men_idx = 7   # ~ 7th column (0-based)
    if women_idx is None:
        women_idx = 10  # two to the right in the 2007-08 sheet

    if header_row is None:
        header_row = 76  # just above first observed data row (~77)

    return GenderColInfo(header_row=header_row, men_col=men_idx, women_col=women_idx)


def _build_name_index(df: pd.DataFrame, header_row: int, name_col: int = 0) -> Dict[str, int]:
    """
    Build a dict mapping lowercased row label -> row index for quick lookup.
    """
    present: Dict[str, int] = {}
    for r in range(header_row + 1, len(df)):
        nm = str(df.iat[r, name_col]).strip()
        if nm and nm.lower() != "nan":
            present[nm.lower()] = r
    return present


def _find_row(label: str, name_index: Dict[str, int]) -> Optional[int]:
    """
    Find the row index for a given label with exact or contains match (lowercased).
    """
    key = label.lower()
    if key in name_index:
        return name_index[key]
    # fuzzy contains
    for k, r in name_index.items():
        if key in k:
            return r
    return None


def extract_period_values(
    xlsx_path: str | Path,
    sheet: str,
    labels: List[str],
    engine: str = "openpyxl",
) -> Tuple[Dict[str, Tuple[float, float]], GenderColInfo]:
    """
    Extract (men_mean, women_mean) for each label from a given period sheet.
    Returns (mapping, gender_info).
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, engine=engine)
    g = detect_gender_cols(raw)
    name_index = _build_name_index(raw, g.header_row, name_col=0)

    out: Dict[str, Tuple[float, float]] = {}
    for label in labels:
        rr = _find_row(label, name_index)
        men_val = women_val = np.nan
        if rr is not None:
            men_val = pd.to_numeric(raw.iat[rr, g.men_col], errors="coerce")
            women_val = pd.to_numeric(raw.iat[rr, g.women_col], errors="coerce")
        out[label] = (men_val, women_val)
    return out, g


def build_tidy_dataframe(xlsx_path: str | Path, engine: str = "openpyxl") -> pd.DataFrame:
    """
    End-to-end preprocessing to produce a tidy DataFrame with columns:
    [period, product, gender, mean, group]
    """
    sheet_map = detect_fah_sheets(xlsx_path, engine=engine)

    records: List[dict] = []
    for period in PERIOD_LABELS:
        sheet = sheet_map[period]

        fruit_vals, _ = extract_period_values(xlsx_path, sheet, FRUITS, engine=engine)
        dairy_vals, _ = extract_period_values(xlsx_path, sheet, DAIRY, engine=engine)

        for label, (m_val, w_val) in fruit_vals.items():
            records.append(dict(period=period, product=label, gender="Men", mean=m_val, group="Fruit"))
            records.append(dict(period=period, product=label, gender="Women", mean=w_val, group="Fruit"))

        for label, (m_val, w_val) in dairy_vals.items():
            records.append(dict(period=period, product=label, gender="Men", mean=m_val, group="Dairy"))
            records.append(dict(period=period, product=label, gender="Women", mean=w_val, group="Dairy"))

    tidy = pd.DataFrame.from_records(records)
    return tidy


def pivot_for_plot(
    tidy: pd.DataFrame,
    group: str,
    gender: str,
    products: List[str],
) -> pd.DataFrame:
    """
    Pivot helper used for plotting: returns a DataFrame indexed by period,
    with one column per product.
    """
    df = tidy[(tidy["group"] == group) & (tidy["gender"] == gender)].copy()
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_LABELS, ordered=True)
    piv = df.pivot_table(index="period", columns="product", values="mean", aggfunc="first")
    return piv.reindex(columns=products)


if __name__ == "__main__":
    # Example: run preprocessing and save a CSV next to the Excel file.
    import argparse
    p = argparse.ArgumentParser(description="Preprocess LAFA FAH sheets (Men/Women) into tidy format.")
    p.add_argument("--xlsx", required=True, help="Path to 'Appendix B (shares).xlsx'")
    p.add_argument("--out", default="lafa_fah_mw_tidy.csv", help="Output CSV path")
    args = p.parse_args()

    tidy = build_tidy_dataframe(args.xlsx)
    tidy.to_csv(args.out, index=False)
    print(f"Saved tidy CSV to: {args.out}")
