import urllib.request
import json
from datetime import date
from dateutil.relativedelta import relativedelta

LOCATION_COORDS = {
    "Chennai":   (13.08, 80.27),
    "Mumbai":    (19.07, 72.87),
    "Delhi":     (28.61, 77.20),
    "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.38, 78.48),
    "Mysuru": (12.30, 76.65),
}

LOCATION_MONTHLY_AVG = {
    "Chennai":   [27,28,30,33,35,35,34,34,33,30,28,26],
    "Mumbai":    [24,25,28,30,31,29,27,27,28,29,27,25],
    "Delhi":     [14,17,23,29,34,34,31,30,29,26,19,14],
    "Bangalore": [22,24,27,29,29,26,24,24,25,24,22,21],
    "Hyderabad": [23,25,29,32,34,32,28,28,28,27,24,22],
}

PROXY = {
    'http':  'http://gateway.schneider.zscaler.net:9480/',
    'https': 'http://gateway.schneider.zscaler.net:9480/',
}


def fetch_monthly_temperature(location: str, billing_period: str) -> float:
    """
    Fetch average temperature for a location and billing month.

    Parameters
    ----------
    location       : city name e.g. "Chennai"
    billing_period : "YYYY-MM" e.g. "2025-07"

    Returns
    -------
    Average temperature in celsius for that month.
    Falls back to historical monthly average if API fails.
    """
    try:
        year, month = billing_period.split("-")
        year, month = int(year), int(month)
    except Exception:
        return 28.0

    # fallback value based on historical monthly average
    month_idx = month - 1
    fallback  = float(LOCATION_MONTHLY_AVG.get(location, [28]*12)[month_idx])

    coords = LOCATION_COORDS.get(location)
    if not coords:
        return fallback

    lat, lon = coords
    start    = date(year, month, 1)
    end      = start + relativedelta(months=1) - relativedelta(days=1)

    # only archive API — billing is always past
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_mean"
        f"&start_date={start.strftime('%Y-%m-%d')}"
        f"&end_date={end.strftime('%Y-%m-%d')}"
        f"&timezone=auto"
    )

    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(PROXY)
        )
        with opener.open(url, timeout=10) as resp:
            data  = json.loads(resp.read())
        temps = [t for t in data["daily"]["temperature_2m_mean"]
                 if t is not None]
        return round(sum(temps)/len(temps), 2) if temps else fallback
    except Exception:
        return fallback


def get_cooling_degree_days(temperature_c: float) -> float:
    """CDD = max(avg_temp - 18, 0)"""
    return max(temperature_c - 18.0, 0.0)


def get_is_summer(temperature_c: float) -> float:
    """
    Returns 1.0 if CDD > 8 (hot month requiring cooling)
    Returns 0.0 otherwise
    """
    return 1.0 if get_cooling_degree_days(temperature_c) > 8 else 0.0


if __name__ == "__main__":
    tests = [
        ("Chennai",   "2025-07"),
        ("Chennai",   "2025-01"),
        ("Delhi",     "2025-06"),
        ("Delhi",     "2025-12"),
        ("Mumbai",    "2025-07"),
        ("Bangalore", "2025-03"),
        ("Hyderabad", "2025-05"),
    ]
    print(f"{'Location':<12} {'Period':<10} {'Temp (C)':>10} "
          f"{'CDD':>8} {'IsSummer':>10}")
    print("-" * 55)
    for loc, period in tests:
        temp      = fetch_monthly_temperature(loc, period)
        cdd       = get_cooling_degree_days(temp)
        is_summer = get_is_summer(temp)
        print(f"{loc:<12} {period:<10} {temp:>10.2f} "
              f"{cdd:>8.2f} {is_summer:>10.1f}")