"""
EcoSEnter Billing Profile Parser — Fixed
==========================================
Fixes:
  1. Detects cumulative vs periodic energy
  2. Subtracts consecutive months for cumulative data
  3. Reads MD correctly — ignores DateTime rows
  4. Displays all extracted values for verification before API call

Usage:
  python parse_and_validate.py --folder "C:/meter_data/Bill" --no-api
  python parse_and_validate.py --folder "C:/meter_data/Bill"
  python parse_and_validate.py --file "Meter.pdf" --no-api
"""

import os
import re
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any


OBIS_FIELDS = {
    "1.0.1.8.0.255":  "totalActiveEnergy",
    "1.0.1.8.1.255":  "rate1Energy",
    "1.0.1.8.2.255":  "rate2Energy",
    "1.0.1.8.3.255":  "rate3Energy",
    "1.0.1.8.4.255":  "rate4Energy",
    "1.0.1.6.0.255":  "peakDemand",
    "1.0.9.6.0.255":  "peakDemandApparent",
    "1.0.9.8.0.255":  "apparentEnergy",
    "1.0.13.0.0.255": "powerFactor",
    "1.0.84.0.0.255": "powerFactorImport",
    "1.0.5.8.0.255":  "reactiveQI",
    "1.0.6.8.0.255":  "reactiveQII",
    "1.0.7.8.0.255":  "reactiveQIII",
    "1.0.8.8.0.255":  "reactiveQIV",
    "0.0.0.1.2.255":  "billingPeriodEnd",
}

# OBIS codes where DateTime rows must be ignored
# These have companion DateTime rows with same OBIS code
MD_OBIS_CODES = {
    "1.0.1.6.0.255", "1.0.9.6.0.255",
    "1.0.1.6.1.255", "1.0.1.6.2.255",
    "1.0.1.6.3.255", "1.0.1.6.4.255",
}


class BillingPeriod:
    def __init__(self):
        self.period:      Optional[str]   = None
        self.readings:    Dict[str,float] = {}
        self.raw_label:   Dict[str,str]   = {}   # parameter description per OBIS

    def get(self, code: str, default: float = 0.0) -> float:
        return float(self.readings.get(code, default))

    def is_valid(self) -> bool:
        return (
            self.period is not None and
            "1.0.1.8.0.255" in self.readings
        )

    def to_api_readings(self) -> List[Dict]:
        return [
            {"obisCode": code, "value": val, "unit": ""}
            for code, val in self.readings.items()
            if code in OBIS_FIELDS and code != "0.0.0.1.2.255"
        ]


def is_datetime_string(value_str: str) -> bool:
    """Check if a value looks like a datetime — not a numeric reading."""
    if not value_str:
        return False
    # datetime patterns: 2025-07-10 11:30:00 or 10/07/2025 etc
    dt_patterns = [
        r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}',
        r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}',
        r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}',
    ]
    for pat in dt_patterns:
        if re.search(pat, str(value_str)):
            return True
    return False


def extract_meter_id(text: str, filename: str) -> str:
    name_match = re.search(r'[-_\s]([A-Z0-9]{5,12})[-_\s]',
                           Path(filename).stem, re.IGNORECASE)
    if name_match:
        cand = name_match.group(1)
        if cand.upper() not in ('DATA','METER','BILLING','ECOENTER',
                                 'ECOSENTER','EXPORT','REPORT'):
            return cand.upper()
    patterns = [
        r'Meter\s+(?:ID|Data)[:\s-]+([A-Z0-9]{5,12})',
        r'Serial[:\s]+([A-Z0-9]{5,12})',
        r'\b([A-Z]\d{7,})\b',
        r'\b([A-Z]{1,2}\d{5,})\b',
        r'\b([A-Z0-9]{6,10})\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def parse_obis_value(value_str: str) -> Optional[float]:
    """Extract numeric value — returns None if it's a datetime or non-numeric."""
    if not value_str:
        return None
    if is_datetime_string(str(value_str)):
        return None
    clean = re.sub(r'[A-Za-z/]+', '', str(value_str)).strip().replace(',', '')
    try:
        return float(clean)
    except ValueError:
        return None


def extract_billing_date(value_str: str) -> Optional[str]:
    if not value_str:
        return None
    patterns = [
        r'(\d{4})-(\d{2})-\d{2}',
        r'(\d{2})/(\d{2})/(\d{4})',
        r'(\d{2})-(\d{2})-(\d{4})',
    ]
    for pat in patterns:
        m = re.search(pat, str(value_str))
        if m:
            groups = m.groups()
            if len(groups) == 2:
                return f"{groups[0]}-{groups[1]}"
            elif len(groups) == 3:
                if len(groups[2]) == 4:
                    return f"{groups[2]}-{groups[1]}"
                else:
                    return f"{groups[0]}-{groups[1]}"
    return None


def derive_meter_type(avg_energy: float) -> tuple:
    if avg_energy < 10_000:
        return "Residential", "LT-1"
    elif avg_energy < 100_000:
        return "Commercial", "LT-2"
    else:
        return "Industrial", "LT-3"


def is_cumulative(periods: List[BillingPeriod]) -> bool:
    """
    Detect if energy values are cumulative register readings.
    If every month is larger than the previous → likely cumulative.
    """
    if len(periods) < 3:
        return False
    sorted_p = sorted(periods, key=lambda p: p.period or "")
    energies  = [p.get("1.0.1.8.0.255") for p in sorted_p]
    # check if strictly increasing
    increasing = sum(1 for i in range(1, len(energies)) if energies[i] > energies[i-1])
    return increasing >= len(energies) - 1  # allow one exception


def subtract_cumulative(periods: List[BillingPeriod]) -> List[BillingPeriod]:
    """
    Convert cumulative register readings to periodic consumption.
    Subtracts each month from the previous.
    Energy OBIS codes that need subtraction:
      1.0.1.8.0.255  total active energy
      1.0.1.8.1-4    rate energies
      1.0.9.8.0.255  apparent energy
    """
    CUMULATIVE_CODES = {
        "1.0.1.8.0.255",
        "1.0.1.8.1.255",
        "1.0.1.8.2.255",
        "1.0.1.8.3.255",
        "1.0.1.8.4.255",
        "1.0.9.8.0.255",
    }

    sorted_p = sorted(periods, key=lambda p: p.period or "")
    result   = []

    for i, period in enumerate(sorted_p):
        new_period = BillingPeriod()
        new_period.period    = period.period
        new_period.raw_label = period.raw_label.copy()

        for code, val in period.readings.items():
            if code in CUMULATIVE_CODES and i > 0:
                prev_val = sorted_p[i-1].get(code, 0.0)
                new_val  = max(val - prev_val, 0.0)
                new_period.readings[code] = round(new_val, 2)
            else:
                new_period.readings[code] = val

        result.append(new_period)

    # first month — keep as is (no previous to subtract from)
    return result


def group_rows_into_periods(rows: List[List[str]]) -> List[BillingPeriod]:
    periods      = []
    current      = BillingPeriod()
    obis_pattern = re.compile(r'\d+\.\d+\.\d+\.\d+\.\d+\.\d+')

    for row in rows:
        obis_code   = None
        value_str   = None
        label_str   = ""

        for i, cell in enumerate(row):
            m = obis_pattern.search(cell)
            if m:
                obis_code = m.group(0)
                remaining = [c for c in row[i+1:] if c.strip()]
                if remaining:
                    value_str = remaining[0]
                # label is usually in column before OBIS code
                if i > 0:
                    label_str = row[i-1].strip()
                break

        if not obis_code:
            full = ' '.join(row)
            if 'Meter Data' in full or 'EcoSEnter' in full:
                if current.is_valid():
                    periods.append(current)
                current = BillingPeriod()
            continue

        if obis_code == "0.0.0.1.2.255":
            period = extract_billing_date(value_str or '')
            if period:
                if current.is_valid() and current.period and \
                   current.period != period:
                    periods.append(current)
                    current = BillingPeriod()
                current.period = period
            continue

        if obis_code in OBIS_FIELDS:
            # for MD codes — skip DateTime rows
            if obis_code in MD_OBIS_CODES:
                if is_datetime_string(value_str or ''):
                    continue  # skip this row entirely

            val = parse_obis_value(value_str or '')
            if val is not None:
                parts   = obis_code.split('.')
                d_value = int(parts[3]) if len(parts) >= 4 else 0
                if d_value in [0, 6, 8]:
                    # only store if not already stored or if this is a better value
                    if obis_code not in current.readings:
                        current.readings[obis_code] = val
                        current.raw_label[obis_code] = label_str

    if current.is_valid():
        periods.append(current)

    return periods


def parse_excel(filepath: str) -> tuple:
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed.")
        sys.exit(1)

    print(f"  Reading: {Path(filepath).name}")
    all_rows = []
    all_text = ""

    try:
        xl = pd.ExcelFile(filepath)
        for sheet in xl.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet,
                               header=None, dtype=str)
            all_text += sheet + "\n"
            for _, row in df.iterrows():
                cells = [str(c).strip() if str(c) != 'nan' else ''
                         for c in row]
                all_rows.append(cells)
                all_text += ' '.join(cells) + "\n"
    except Exception as e:
        print(f"  Error: {e}")
        return "UNKNOWN", []

    meter_id = extract_meter_id(all_text, Path(filepath).name)
    periods  = group_rows_into_periods(all_rows)
    return meter_id, periods


def parse_pdf(filepath: str) -> tuple:
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not installed.")
        sys.exit(1)

    print(f"  Reading: {Path(filepath).name}")
    all_text = ""
    all_rows = []

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text += text + "\n"
            for table in page.extract_tables():
                for row in table:
                    if row:
                        all_rows.append([str(c or '').strip() for c in row])

    meter_id = extract_meter_id(all_text, Path(filepath).name)
    periods  = group_rows_into_periods(all_rows)
    return meter_id, periods


def parse_csv(filepath: str) -> tuple:
    print(f"  Reading: {Path(filepath).name}")
    all_rows = []
    all_text = ""

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_text += line + "\n"
            parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)
            parts = [p.strip().strip('"') for p in parts]
            all_rows.append(parts)

    meter_id = extract_meter_id(all_text, Path(filepath).name)
    periods  = group_rows_into_periods(all_rows)
    return meter_id, periods


def parse_folder(folder_path: str) -> tuple:
    folder = Path(folder_path)
    files  = sorted(
        list(folder.glob("*.pdf")) + list(folder.glob("*.PDF")) +
        list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")) +
        list(folder.glob("*.csv"))
    )

    if not files:
        raise ValueError(f"No files found in: {folder_path}")

    print(f"  Found {len(files)} files")

    all_periods = []
    meter_id    = "UNKNOWN"

    for f in files:
        ext = f.suffix.lower()
        try:
            if ext == ".pdf":
                mid, periods = parse_pdf(str(f))
            elif ext in (".xlsx", ".xls"):
                mid, periods = parse_excel(str(f))
            else:
                mid, periods = parse_csv(str(f))

            if mid and mid != "UNKNOWN" and meter_id == "UNKNOWN":
                meter_id = mid

            all_periods.extend(periods)
        except Exception as e:
            print(f"    WARNING: {f.name}: {e}")

    seen = {}
    for p in all_periods:
        if p.period:
            seen[p.period] = p

    return meter_id, list(seen.values())


def display_all_values(periods: List[BillingPeriod],
                       cumulative: bool,
                       meter_id: str):
    """
    Display all extracted values per month for verification.
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  EXTRACTED VALUES — Meter {meter_id}")
    print(f"  Data type: {'CUMULATIVE (subtracted)' if cumulative else 'PERIODIC'}")
    print(sep)

    sorted_p = sorted(periods, key=lambda p: p.period or "")

    for p in sorted_p:
        print(f"\n  Period: {p.period}")
        print(f"  {'─'*60}")
        print(f"  {'Parameter':<40} {'OBIS Code':<18} {'Value':>15}")
        print(f"  {'─'*40} {'─'*18} {'─'*15}")

        display_map = [
            ("Total Active Energy",    "1.0.1.8.0.255",  "Wh"),
            ("Rate 1 Energy",          "1.0.1.8.1.255",  "Wh"),
            ("Rate 2 Energy",          "1.0.1.8.2.255",  "Wh"),
            ("Rate 3 Energy",          "1.0.1.8.3.255",  "Wh"),
            ("Rate 4 Energy",          "1.0.1.8.4.255",  "Wh"),
            ("Peak Demand (MD)",       "1.0.1.6.0.255",  "W"),
            ("Apparent Energy",        "1.0.9.8.0.255",  "VAh"),
            ("Power Factor Avg",       "1.0.13.0.0.255", ""),
            ("Power Factor Import",    "1.0.84.0.0.255", ""),
            ("Reactive Energy QI",     "1.0.5.8.0.255",  "VArh"),
            ("Reactive Energy QIV",    "1.0.8.8.0.255",  "VArh"),
        ]

        for label, code, unit in display_map:
            if code in p.readings:
                val = p.readings[code]
                print(f"  {label:<40} {code:<18} {val:>12,.2f} {unit}")
            else:
                print(f"  {label:<40} {code:<18} {'—':>15}")

    print(f"\n{sep}")


def build_payload(meter_id, periods, location="Chennai"):
    if not periods:
        raise ValueError("No valid billing periods found")

    periods.sort(key=lambda p: p.period or "")
    current  = periods[-1]
    history  = periods[:-1]

    energies   = [p.get("1.0.1.8.0.255") for p in history
                  if p.get("1.0.1.8.0.255") > 0]
    avg_energy = sum(energies)/len(energies) if energies else \
                 current.get("1.0.1.8.0.255")
    meter_type, tariff = derive_meter_type(avg_energy)

    return {
        "utilityId":       "default",
        "meterId":         meter_id,
        "meterType":       meter_type,
        "tariffSlab":      tariff,
        "location":        location,
        "billingPeriod":   current.period or "unknown",
        "historyMonths":   len(history),
        "currentReadings": current.to_api_readings(),
        "history": [
            {"billingPeriod": p.period, "readings": p.to_api_readings()}
            for p in history
        ],
    }


def call_api(payload, api_url):
    url  = f"{api_url.rstrip('/')}/api/validate"
    data = json.dumps(payload).encode('utf-8')
    req  = urllib.request.Request(
        url, data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API error {e.code}: {e.read().decode()}")


def print_result(result):
    sev   = result.get('severity','?')
    score = result.get('riskScore',0)
    sep   = "=" * 60
    color = {'HIGH':'\033[91m','MEDIUM':'\033[93m','LOW':'\033[92m'}.get(sev,'')
    reset = '\033[0m'

    print(f"\n{sep}")
    print(f"  Validation Result — {result.get('meterId')}")
    print(sep)
    print(f"  Billing period   : {result.get('billingPeriod')}")
    print(f"\n  {color}Risk score  : {score}{reset}")
    print(f"  {color}Severity    : {sev}{reset}")
    print(f"  Anomaly type     : {result.get('anomalyType','?')}")
    print(f"\n  Actual Wh        : {result.get('actualWh',0):,.0f} Wh")
    print(f"  Predicted Wh     : {result.get('predictedWh',0):,.0f} Wh")
    print(f"  Deviation        : {result.get('deviationPct',0):.1f}%")
    print(f"  Historical avg   : {result.get('historicalAvgWh',0):,.0f} Wh")
    print(f"\n  NTL suspected    : {result.get('ntlSuspected')}")
    print(f"  Pre-invoice flag : {result.get('preInvoiceFlag')}")
    print(f"  Revenue at risk  : Rs {result.get('revenueAtRiskINR',0):,.2f}")
    print(f"\n  Model confidence : {result.get('modelConfidence')} "
          f"({result.get('historyMonths')} months)")

    flags = [k for k,v in (result.get('ruleFlags') or {}).items() if v]
    if flags:
        print(f"\n  Rules fired      : {', '.join(flags)}")

    sc = result.get('scoreComponents') or {}
    print(f"\n  Score breakdown:")
    print(f"    IF       : {sc.get('isolation_forest',0):.1f} pts")
    print(f"    TFT      : {sc.get('tft_deviation',0)} pts")
    print(f"    Rules    : {sc.get('rules',0)} pts")
    print(f"    Total    : {score}")
    print(f"\n  Recommendation   : {color}{result.get('recommendation')}{reset}")
    print(sep)

    if result.get('reports',{}).get('excel'):
        print(f"\n  Reports:")
        print(f"    Excel : http://127.0.0.1:8000{result['reports']['excel']}")
        print(f"    PDF   : http://127.0.0.1:8000{result['reports']['pdf']}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse EcoSEnter billing profile and validate with BVEngine'
    )
    parser.add_argument('--file',     default=None)
    parser.add_argument('--folder',   default=None)
    parser.add_argument('--api',      default='http://127.0.0.1:8000')
    parser.add_argument('--location', default='Chennai')
    parser.add_argument('--no-api',   action='store_true')
    parser.add_argument('--save',     action='store_true')
    args = parser.parse_args()

    if not args.file and not args.folder:
        print("ERROR: Provide --file or --folder")
        parser.print_help()
        sys.exit(1)

    # ── Parse ──────────────────────────────────────────────────────────────────
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"ERROR: Folder not found: {args.folder}")
            sys.exit(1)
        print(f"\nFolder: {args.folder}")
        meter_id, periods = parse_folder(args.folder)
    else:
        if not os.path.exists(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        ext = Path(args.file).suffix.lower()
        print(f"\nFile: {args.file}")
        if ext == '.pdf':
            meter_id, periods = parse_pdf(args.file)
        elif ext in ('.xlsx','.xls'):
            meter_id, periods = parse_excel(args.file)
        else:
            meter_id, periods = parse_csv(args.file)

    if not periods:
        print("ERROR: No billing periods found.")
        sys.exit(1)

    # ── Detect cumulative ──────────────────────────────────────────────────────
    cumulative = is_cumulative(periods)
    print(f"\n  Meter ID       : {meter_id}")
    print(f"  Periods found  : {len(periods)}")
    print(f"  Data type      : {'CUMULATIVE — will subtract months' if cumulative else 'PERIODIC — values used as-is'}")

    if cumulative:
        print(f"\n  Applying cumulative subtraction...")
        periods = subtract_cumulative(periods)
        print(f"  Done.")

    # ── Display all values ─────────────────────────────────────────────────────
    display_all_values(periods, cumulative, meter_id)

    print(f"\n  Please verify the values above against your Excel files.")
    print(f"  If correct, proceed. If not, check the data type detection.")

    if args.no_api:
        print(f"\n  --no-api set. Skipping API call.")
        return

    # ── Build payload ──────────────────────────────────────────────────────────
    if len(periods) < 2:
        print("ERROR: Need at least 2 periods for validation.")
        sys.exit(1)

    try:
        payload = build_payload(meter_id, periods, args.location)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\n  Meter type     : {payload['meterType']} ({payload['tariffSlab']})")
    print(f"  Current period : {payload['billingPeriod']}")
    print(f"  History months : {payload['historyMonths']}")

    if args.save:
        out = f"{meter_id}_payload.json"
        with open(out,'w') as f:
            json.dump(payload, f, indent=2)
        print(f"  Payload saved  : {out}")

    print(f"\n  Calling API...")
    try:
        result = call_api(payload, args.api)
        print_result(result)
        res_path = f"{meter_id}_result.json"
        with open(res_path,'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result saved: {res_path}")
    except RuntimeError as e:
        print(f"\n  API Error: {e}")
        print("  Make sure API is running:")
        print("  python -m uvicorn api.main:app --reload --port 8000")


if __name__ == '__main__':
    main()