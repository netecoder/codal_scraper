#!/usr/bin/env python
"""
Fetch TSETMC shareholder + price data for Codal board-member dataset,
using a manual symbol correction file with a 'symbol_true' column.

Input 1 (board members):
    - output/board_members_1394_1404_from_urls.csv
      must have at least:
        * company: "شرکت سیمان خزر - نماد: سخزر"
                   or "شرکت سیمان لار سبزوار - نماد: سبزوا(ذسبزوار)"
                   or "شرکت آ.س.پ - نماد: آ س پ(ثاسپ)"
        * assembly_date: Jalali like "14021223" or "1402/12/23"

Input 2 (manual corrections):
    - output/tsetmc_unresolved_symbols.csv
      with columns at least:
        * symbol       (original unresolved candidate)
        * symbol_true  (manually corrected TSETMC symbol name)
      Example row:
        symbol, pair_count, symbol_true
        بتجارت, 3, بمولد

For each (company, assembly_date):

    1) Extract ALL symbol candidates from `company`:
        - the string after "نماد" and ":" up to the first "("  (pre_paren)
        - the string inside the first "( ... )"                 (paren_content)

    2) For each candidate, if it appears in the manual corrections file,
       replace it by its 'symbol_true'.

    3) For each candidate symbol (in order):
        - resolve Instrument via TSETMC
        - fetch info() once (zTitad, insCode, sector code/name)
        - convert Jalali assembly_date -> Gregorian date g_target
        - for g = g_target, g+1, g+2:
            * dEven = YYYYMMDD (Gregorian)
            * ins.on_date(dEven).holders()
              - if non-empty, use that date
            * ins.on_date(dEven).closing_price() -> pClosing

    4) Emit one row per shareholder with:
         - company, symbol (effective candidate)
         - assembly_date_jalali, effective_gregorian_date, effective_dEven
         - insCode, isic_code (sector.cSecVal), sector_name, zTitad, pClosing
         - shareHolderName, perOfShares, shareHolderShareID

If no symbol candidate resolves to an Instrument -> goes to
tsetmc_unresolved_symbols_remaining.csv.

If an Instrument is resolved but no holders data in the 0..2 day window,
that attempt goes to tsetmc_no_data_assemblies.csv.

Outputs:
    - output/tsetmc_shareholders_1394_1404.csv
    - output/tsetmc_unresolved_symbols_remaining.csv
    - output/tsetmc_no_data_assemblies.csv

Requirements:
    pip install "tsetmc>=2.0" jdatetime pandas
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date as Date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from jdatetime import date as JDate
from tsetmc.instruments import Instrument


# ---------------------- Configuration ---------------------- #

BASE_DIR = Path("output")

INPUT_CSV = BASE_DIR / "board_members_1394_1404_from_urls.csv"
OUTPUT_CSV = BASE_DIR / "tsetmc_shareholders_1394_1404.csv"

# Manual corrections file with columns: symbol, symbol_true
SYMBOL_CORRECTIONS_CSV = BASE_DIR / "tsetmc_unresolved_symbols.csv"

UNRESOLVED_OUT_CSV = BASE_DIR / "tsetmc_unresolved_symbols_remaining.csv"
NO_DATA_ASSEMBLIES_CSV = BASE_DIR / "tsetmc_no_data_assemblies.csv"

COMPANY_COLUMN = "company"
ASSEMBLY_DATE_COLUMN_CANDIDATES = ["assembly_date", "AssemblyDate", "assemblyDate"]

MAX_CONCURRENT_ROWS = 8
REQUEST_RETRIES = 3
REQUEST_RETRY_BASE_DELAY = 1.5  # seconds
MAX_FORWARD_DAYS = 2            # assembly date + 0,1,2


# ---------------------- Logging ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("tsetmc_shareholders")


# ---------------------- Failure registries & caches ---------------------- #

UNRESOLVED_SYMBOLS: set = set()        # symbols that couldn't map to any Instrument
NO_DATA_ASSEMBLIES: List[Dict[str, Any]] = []  # attempts with no holders in 0..2d window

INSTRUMENT_CACHE: Dict[str, Optional[Instrument]] = {}
INFO_CACHE: Dict[str, Dict[str, Any]] = {}
SYMBOL_CORRECTIONS: Dict[str, str] = {}  # normalized original -> cleaned symbol_true


# ---------------------- Digit & text normalization ---------------------- #

FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(value) -> str:
    """Convert Persian/Arabic digits to ASCII digits and return as string."""
    s = str(value)
    s = s.translate(FA_DIGITS)
    s = s.translate(AR_DIGITS)
    return s


def _strip_rtl_noise(text: str) -> str:
    """Remove zero-width and common RTL junk characters."""
    if not isinstance(text, str):
        return text
    return text.replace("\u200c", "").replace("\u200f", "").strip()


def _normalize_symbol_key(s: str) -> str:
    """Normalize symbol text for dictionary keys."""
    if s is None:
        return ""
    s = normalize_digits(str(s))
    s = _strip_rtl_noise(s)
    # collapse internal whitespace
    s = " ".join(s.split())
    return s


# ---------------------- Symbol and date helpers ---------------------- #

def extract_symbol_candidates(company_str: str) -> List[str]:
    """
    Extract ALL possible TSETMC symbol candidates from a 'company' column value.

    We consider BOTH:
        - the string after "نماد" and ":" up to the first "("
        - the string inside the first "( ... )"

    Examples:
        "شرکت سیمان خزر - نماد: سخزر"
            -> ["سخزر"]

        "شرکت آ.س.پ - نماد: آ س پ(ثاسپ)"
            -> ["ثاسپ", "آ س پ"]

        "شرکت سیمان لار سبزوار - نماد: سبزوا(ذسبزوار)"
            -> ["ذسبزوار", "سبزوا"]
    """
    if not isinstance(company_str, str):
        return []

    text = _strip_rtl_noise(company_str)

    idx = text.find("نماد")
    if idx == -1:
        return []

    # Rest after 'نماد'
    rest = text[idx + len("نماد") :]
    # Strip colon variants and whitespace
    rest = rest.lstrip(" :：").strip()
    if not rest:
        return []

    # Locate first parentheses
    open_idx = rest.find("(")
    close_idx = rest.find(")", open_idx + 1) if open_idx != -1 else -1

    # Part before '('
    if open_idx != -1:
        pre_paren_raw = rest[:open_idx]
    else:
        pre_paren_raw = rest

    # Clean pre-paren: cut at first '-', '،', ',', '|'
    cut = len(pre_paren_raw)
    for ch in ("-", "،", ",", "|"):
        pos = pre_paren_raw.find(ch)
        if pos != -1:
            cut = min(cut, pos)
    pre_paren = pre_paren_raw[:cut].strip()

    # Content inside first '( ... )'
    paren_content = ""
    if open_idx != -1 and close_idx != -1 and close_idx > open_idx:
        paren_content = rest[open_idx + 1 : close_idx].strip()

    candidates: List[str] = []

    def _clean_sym(s: str) -> str:
        s = _strip_rtl_noise(s)
        s = " ".join(s.split())
        return s.strip("()[]،,;")

    # Prefer text inside parentheses first (often the real ticker),
    # but also try the text before '('.
    if paren_content:
        c = _clean_sym(paren_content)
        if c:
            candidates.append(c)

    if pre_paren:
        c = _clean_sym(pre_paren)
        if c and c not in candidates:
            candidates.append(c)

    return candidates


def jalali_str_to_gregorian_date(jalali) -> Optional[Date]:
    """
    Convert a Jalali date (string or int) to a Gregorian date.

    Accepts:
        - 1402/12/23
        - 1402-12-23
        - 14021223
        - same with Persian/Arabic digits

    Returns:
        datetime.date or None if it cannot be parsed.
    """
    if jalali is None:
        return None
    if isinstance(jalali, float) and math.isnan(jalali):
        return None

    s = normalize_digits(jalali).strip()
    if not s:
        return None

    # Remove separators; work with compact YYYYMMDD
    s = s.replace("/", "").replace("-", "")
    if len(s) != 8 or not s.isdigit():
        logger.warning("Cannot parse Jalali date: %r (normalized: %r)", jalali, s)
        return None

    try:
        jy = int(s[0:4])
        jm = int(s[4:6])
        jd = int(s[6:8])

        jdate = JDate(jy, jm, jd)
        gdate = jdate.togregorian()
        return Date(gdate.year, gdate.month, gdate.day)
    except Exception as exc:
        logger.warning("Error converting Jalali date %r -> Gregorian: %s", jalali, exc)
        return None


def gregorian_to_deven(d: Date) -> int:
    """Convert Gregorian date to dEven integer YYYYMMDD."""
    return d.year * 10000 + d.month * 100 + d.day


# ---------------------- DataFrame helpers ---------------------- #

def normalize_holder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize holder DataFrame columns to:
        - shareHolderName
        - perOfShares
        - shareHolderShareID
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["shareHolderName", "perOfShares", "shareHolderShareID"]
        )

    col_map: Dict[str, str] = {}
    for col in df.columns:
        norm = str(col).replace(" ", "").replace("_", "").lower()

        if norm in {
            "shareholdername",
            "سهامداردارنده",
            "سهامدار/دارنده",
            "holdername",
            "name",
        }:
            col_map["shareHolderName"] = col
        elif norm in {"perofshares", "percent", "درصدسهام", "درصدسهم"}:
            col_map["perOfShares"] = col
        elif norm in {"shareholdershareid", "idcisin", "id"}:
            col_map["shareHolderShareID"] = col

    out = pd.DataFrame()
    for target_col in ["shareHolderName", "perOfShares", "shareHolderShareID"]:
        src_col = col_map.get(target_col)
        out[target_col] = df[src_col] if src_col is not None else None

    return out


async def call_with_retries(
    coro_factory,
    retries: int = REQUEST_RETRIES,
    base_delay: float = REQUEST_RETRY_BASE_DELAY,
):
    """
    Run an async callable with simple exponential backoff.
    `coro_factory` is a zero-arg callable that returns a coroutine.

    We treat IndexError as *non-retriable* because in tsetmc this often
    means 'instrument/data not found' rather than a transient network issue.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except IndexError as exc:
            logger.debug("Non-retriable IndexError in call_with_retries: %s", exc)
            last_exc = exc
            break
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                logger.error("Failed after %d attempts: %s", attempt, exc)
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Error in attempt %d/%d: %s – retrying in %.1fs",
                attempt,
                retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exc:
        raise last_exc
    return None


@dataclass
class AssemblyRow:
    company: str
    assembly_date_jalali: str
    symbol_candidates: List[str]


# ---------------------- Symbol corrections ---------------------- #

def load_symbol_corrections() -> None:
    """
    Load manual symbol corrections from SYMBOL_CORRECTIONS_CSV into
    SYMBOL_CORRECTIONS dict: normalized original -> cleaned symbol_true.
    """
    if not SYMBOL_CORRECTIONS_CSV.exists():
        logger.warning("Symbol corrections CSV not found: %s", SYMBOL_CORRECTIONS_CSV)
        return

    try:
        df = pd.read_csv(SYMBOL_CORRECTIONS_CSV)
    except Exception as exc:
        logger.error("Failed to read symbol corrections CSV %s: %s",
                     SYMBOL_CORRECTIONS_CSV, exc)
        return

    if "symbol" not in df.columns or "symbol_true" not in df.columns:
        logger.error(
            "Symbol corrections CSV must have 'symbol' and 'symbol_true' columns."
        )
        return

    count = 0
    for _, row in df.iterrows():
        raw_sym = row.get("symbol")
        raw_true = row.get("symbol_true")
        if pd.isna(raw_sym) or pd.isna(raw_true):
            continue

        key = _normalize_symbol_key(raw_sym)
        val = _normalize_symbol_key(raw_true)
        if not key or not val:
            continue

        SYMBOL_CORRECTIONS[key] = val
        count += 1

    logger.info("Loaded %d symbol corrections from %s", count, SYMBOL_CORRECTIONS_CSV)


def apply_symbol_corrections(candidates: List[str]) -> List[str]:
    """
    Apply manual corrections to a list of symbol candidates.
    If a candidate exists as a key in SYMBOL_CORRECTIONS, replace it
    with symbol_true; otherwise keep it. Deduplicate while preserving order.
    """
    out: List[str] = []
    seen: set = set()

    for c in candidates:
        key = _normalize_symbol_key(c)
        corrected = SYMBOL_CORRECTIONS.get(key, c)
        corrected_norm = _normalize_symbol_key(corrected)
        if not corrected_norm:
            continue
        if corrected_norm in seen:
            continue
        seen.add(corrected_norm)
        out.append(corrected)

    return out


# ---------------------- Core TSETMC helpers ---------------------- #

async def fetch_snapshot_for_date_window(
    inst: Instrument,
    g_target: Date,
    symbol_label: str,
) -> Tuple[Optional[Date], Optional[int], Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    For a given instrument & target Gregorian date, try:
        d = target, target+1, target+2
        dEven = YYYYMMDD
        - await inst.on_date(dEven).holders()
        - if non-empty: await inst.on_date(dEven).closing_price()

    Returns:
        (effective_g_date, effective_dEven, holders_df, price_info_dict)
    """
    for offset in range(0, MAX_FORWARD_DAYS + 1):
        g_candidate = g_target + timedelta(days=offset)
        d_even = gregorian_to_deven(g_candidate)

        # holders()
        try:
            async def holders_coro():
                snap = inst.on_date(d_even)
                return await snap.holders()

            raw_holders = await call_with_retries(holders_coro)
        except Exception as exc:
            logger.debug(
                "holders() failed for symbol %s at dEven=%d: %s",
                symbol_label,
                d_even,
                exc,
            )
            continue

        if raw_holders is None or getattr(raw_holders, "empty", False):
            # No holder data for that date
            continue

        # Normalize and fetch price info for the same date
        holders_df = normalize_holder_columns(raw_holders)

        try:
            async def price_coro():
                snap = inst.on_date(d_even)
                return await snap.closing_price()

            price_info = await call_with_retries(price_coro)
        except Exception as exc:
            logger.warning(
                "closing_price() failed for symbol %s at dEven=%d: %s",
                symbol_label,
                d_even,
                exc,
            )
            price_info = {}

        return g_candidate, d_even, holders_df, price_info

    # No usable snapshot within the allowed window
    return None, None, None, None


async def resolve_instrument(symbol: str) -> Optional[Instrument]:
    """
    Resolve a TSETMC Instrument for a given symbol string (candidate).

    We **do not** drop first letters (ذ, و, ن, etc.) – we use the candidate
    exactly as parsed, and try:
        1) Instrument.from_l18(symbol_clean)
        2) Instrument.from_search(symbol_clean)
    """
    symbol_clean = _normalize_symbol_key(symbol)

    # 1) from_l18
    try:
        inst = await Instrument.from_l18(symbol_clean)
        return inst
    except Exception as exc1:
        logger.warning(
            "Instrument.from_l18 failed for symbol %s: %s – trying from_search",
            symbol_clean,
            exc1,
        )

    # 2) from_search
    try:
        inst = await Instrument.from_search(symbol_clean)
        return inst
    except Exception as exc2:
        logger.error(
            "Failed to resolve instrument for symbol %s via from_search: %s",
            symbol_clean,
            exc2,
        )
        UNRESOLVED_SYMBOLS.add(symbol_clean)
        return None


async def get_instrument(symbol: str) -> Optional[Instrument]:
    """Return Instrument for symbol, using cache."""
    key = _normalize_symbol_key(symbol)
    if key in INSTRUMENT_CACHE:
        return INSTRUMENT_CACHE[key]

    inst = await resolve_instrument(symbol)
    INSTRUMENT_CACHE[key] = inst
    return inst


async def get_info(inst: Instrument, symbol_key: str) -> Dict[str, Any]:
    """Return info() for an instrument, cached by symbol_key."""
    key = _normalize_symbol_key(symbol_key)
    if key in INFO_CACHE:
        return INFO_CACHE[key]

    try:
        info = await call_with_retries(lambda: inst.info())
    except Exception as exc:
        logger.error("Failed to fetch info() for symbol %s: %s", symbol_key, exc)
        info = {}

    INFO_CACHE[key] = info or {}
    return INFO_CACHE[key]


# ---------------------- Per-row processing ---------------------- #

async def process_row(row: AssemblyRow) -> List[Dict[str, Any]]:
    """
    For a single (company, assembly_date, candidate symbols) row:

        - Try each candidate symbol (in order) against TSETMC.
        - For the first candidate that yields non-empty holders data
          in the date window, emit one row per shareholder.
        - If none works, record in NO_DATA_ASSEMBLIES and emit a
          single "metadata-only" row (with shareholder columns = None).
    """
    results: List[Dict[str, Any]] = []
    candidates = [c for c in row.symbol_candidates if c]

    if not candidates:
        NO_DATA_ASSEMBLIES.append(
            {
                "company": row.company,
                "symbols_tried": "",
                "assembly_date_jalali": row.assembly_date_jalali,
                "reason": "no_symbol_candidates",
            }
        )
        return results

    g_target = jalali_str_to_gregorian_date(row.assembly_date_jalali)
    if not g_target:
        logger.warning(
            "Skipping row for company %s, symbols %s: cannot parse assembly_date %r",
            row.company,
            "|".join(candidates),
            row.assembly_date_jalali,
        )
        NO_DATA_ASSEMBLIES.append(
            {
                "company": row.company,
                "symbols_tried": "|".join(candidates),
                "assembly_date_jalali": row.assembly_date_jalali,
                "reason": "invalid_jalali_date",
            }
        )
        # Emit one meta row with no shareholders
        results.append(
            {
                "company": row.company,
                "symbol": None,
                "assembly_date_jalali": row.assembly_date_jalali,
                "effective_gregorian_date": None,
                "effective_dEven": None,
                "insCode": None,
                "isic_code": None,
                "sector_name": None,
                "zTitad": None,
                "pClosing": None,
                "shareHolderName": None,
                "perOfShares": None,
                "shareHolderShareID": None,
            }
        )
        return results

    last_info: Dict[str, Any] = {}
    last_symbol_used: Optional[str] = None

    # Try each candidate
    for cand in candidates:
        inst = await get_instrument(cand)
        if inst is None:
            # This candidate could not be resolved to an Instrument
            continue

        info = await get_info(inst, cand)
        last_info = info
        last_symbol_used = cand

        sector = info.get("sector") or {}
        isic_code = (sector.get("cSecVal") or "").strip() or None
        sector_name = sector.get("lSecVal") or None
        z_titad = info.get("zTitad")
        ins_code = info.get("insCode")

        g_eff, d_even_eff, holders_df, price_info = await fetch_snapshot_for_date_window(
            inst, g_target, cand
        )

        if g_eff is None or d_even_eff is None or holders_df is None:
            # No data for this candidate; try next candidate
            continue

        # Extract pClosing (if provided)
        p_closing = None
        if isinstance(price_info, dict):
            p_closing = price_info.get("pClosing") or price_info.get("pc")

        # Emit one row per shareholder
        for _, hrow in holders_df.iterrows():
            results.append(
                {
                    "company": row.company,
                    "symbol": cand,
                    "assembly_date_jalali": row.assembly_date_jalali,
                    "effective_gregorian_date": g_eff.isoformat(),
                    "effective_dEven": d_even_eff,
                    "insCode": ins_code,
                    "isic_code": isic_code,
                    "sector_name": sector_name,
                    "zTitad": z_titad,
                    "pClosing": p_closing,
                    "shareHolderName": hrow.get("shareHolderName"),
                    "perOfShares": hrow.get("perOfShares"),
                    "shareHolderShareID": hrow.get("shareHolderShareID"),
                }
            )

        # We found a working candidate; stop trying others
        return results

    # If we reach here, no candidate produced holders data
    NO_DATA_ASSEMBLIES.append(
        {
            "company": row.company,
            "symbols_tried": "|".join(candidates),
            "assembly_date_jalali": row.assembly_date_jalali,
            "reason": "no_holders_for_any_candidate",
        }
    )

    # Emit one "meta" row so we know the attempt exists in the main CSV
    sector = last_info.get("sector") or {}
    isic_code = (sector.get("cSecVal") or "").strip() or None
    sector_name = sector.get("lSecVal") or None
    z_titad = last_info.get("zTitad")
    ins_code = last_info.get("insCode")

    # If nothing ever resolved, last_symbol_used may be None
    symbol_for_meta = last_symbol_used or (candidates[0] if candidates else None)

    results.append(
        {
            "company": row.company,
            "symbol": symbol_for_meta,
            "assembly_date_jalali": row.assembly_date_jalali,
            "effective_gregorian_date": None,
            "effective_dEven": None,
            "insCode": ins_code,
            "isic_code": isic_code,
            "sector_name": sector_name,
            "zTitad": z_titad,
            "pClosing": None,
            "shareHolderName": None,
            "perOfShares": None,
            "shareHolderShareID": None,
        }
    )
    return results


# ---------------------- Main orchestration ---------------------- #

async def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    logger.info("Reading input CSV: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    if COMPANY_COLUMN not in df.columns:
        raise SystemExit(f"Expected a '{COMPANY_COLUMN}' column in input CSV.")

    # Detect assembly date column
    assembly_col = None
    for cand in ASSEMBLY_DATE_COLUMN_CANDIDATES:
        if cand in df.columns:
            assembly_col = cand
            break
    if assembly_col is None:
        raise SystemExit(
            f"Could not find assembly date column; tried {ASSEMBLY_DATE_COLUMN_CANDIDATES}"
        )

    logger.info("Using assembly date column: %s", assembly_col)

    # Load manual symbol corrections
    load_symbol_corrections()
    logger.info("Symbol corrections loaded: %d entries", len(SYMBOL_CORRECTIONS))

    # (company, assembly_date) pairs
    df_pairs = df[[COMPANY_COLUMN, assembly_col]].dropna().drop_duplicates()
    logger.info(
        "Found %d unique (company, %s) pairs.",
        len(df_pairs),
        assembly_col,
    )

    # Extract raw symbol candidates from company string
    df_pairs["symbol_candidates_raw"] = df_pairs[COMPANY_COLUMN].apply(
        extract_symbol_candidates
    )

    # Apply corrections to get final candidate list
    df_pairs["symbol_candidates"] = df_pairs["symbol_candidates_raw"].apply(
        apply_symbol_corrections
    )

    before_filter = len(df_pairs)
    df_pairs = df_pairs[df_pairs["symbol_candidates"].map(lambda lst: bool(lst))]
    dropped = before_filter - len(df_pairs)

    logger.info(
        "Kept %d pairs with at least one (possibly corrected) symbol candidate (dropped %d).",
        len(df_pairs),
        dropped,
    )

    rows: List[AssemblyRow] = [
        AssemblyRow(
            company=row[COMPANY_COLUMN],
            assembly_date_jalali=row[assembly_col],
            symbol_candidates=row["symbol_candidates"],
        )
        for _, row in df_pairs.iterrows()
    ]

    logger.info("Processing %d assembly rows via TSETMC...", len(rows))

    sem = asyncio.Semaphore(MAX_CONCURRENT_ROWS)

    async def worker(r: AssemblyRow) -> List[Dict[str, Any]]:
        async with sem:
            return await process_row(r)

    tasks = [worker(r) for r in rows]
    all_chunks = await asyncio.gather(*tasks, return_exceptions=False)

    flat_rows: List[Dict[str, Any]] = []
    for chunk in all_chunks:
        flat_rows.extend(chunk)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Save main shareholders data --- #
    if not flat_rows:
        logger.warning("No data collected from TSETMC – nothing to write for main CSV.")
    else:
        out_df = pd.DataFrame(flat_rows)
        out_df.sort_values(
            by=["symbol", "assembly_date_jalali", "shareHolderName"],
            inplace=True,
            na_position="last",
        )
        out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        logger.info("Written %d rows to %s", len(out_df), OUTPUT_CSV.resolve())

    # --- Save unresolved symbols (after corrections) --- #
    if UNRESOLVED_SYMBOLS:
        rows_counts = []
        for sym in sorted(UNRESOLVED_SYMBOLS):
            count = df_pairs["symbol_candidates"].apply(
                lambda lst, s=sym: _normalize_symbol_key(s) in
                {_normalize_symbol_key(x) for x in lst}
            ).sum()
            rows_counts.append({"symbol": sym, "pair_count": int(count)})

        counts_df = pd.DataFrame(rows_counts)
        UNRESOLVED_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(UNRESOLVED_OUT_CSV, index=False, encoding="utf-8-sig")
        logger.info(
            "Written %d unresolved symbols to %s",
            len(counts_df),
            UNRESOLVED_OUT_CSV.resolve(),
        )
    else:
        logger.info(
            "No unresolved symbols – nothing to write to %s",
            UNRESOLVED_OUT_CSV,
        )

    # --- Save no-data assemblies --- #
    if NO_DATA_ASSEMBLIES:
        no_data_df = pd.DataFrame(NO_DATA_ASSEMBLIES)
        no_data_df.sort_values(
            by=["company", "assembly_date_jalali"],
            inplace=True,
            na_position="last",
        )
        NO_DATA_ASSEMBLIES_CSV.parent.mkdir(parents=True, exist_ok=True)
        no_data_df.to_csv(NO_DATA_ASSEMBLIES_CSV, index=False, encoding="utf-8-sig")
        logger.info(
            "Written %d no-data assembly attempts to %s",
            len(no_data_df),
            NO_DATA_ASSEMBLIES_CSV.resolve(),
        )
    else:
        logger.info(
            "No no-data assemblies – nothing to write to %s",
            NO_DATA_ASSEMBLIES_CSV,
        )


if __name__ == "__main__":
    asyncio.run(main())
