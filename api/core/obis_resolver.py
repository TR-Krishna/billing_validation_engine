"""
OBIS Resolver
=============
Converts raw HES OBIS key-value pairs into clean named fields
that the feature engine and rules engine can work with.

This is the plugin model's core piece.
Different utilities use different OBIS codes for the same parameter.
The config file maps codes to field names.
The resolver converts the raw payload using that mapping.

Example:
  HES sends:
    { "obisCode": "1.0.1.8.0.255", "value": 15820, "unit": "Wh" }

  Config maps:
    "totalActiveEnergy": "1.0.1.8.0.255"

  Resolver returns:
    { "totalActiveEnergy": 15820 }

This means the engine never hardcodes OBIS codes.
To support a new utility just add a new config file.
Zero code changes needed.
"""

from typing import List, Dict, Any, Optional
from api.models.request import OBISReading, HistoricalMonth


class OBISResolver:
    """
    Resolves OBIS codes to meaningful field names using config mapping.
    One instance per request — created fresh for each API call.
    """

    def __init__(self, obis_map: Dict[str, str],
                 thresholds: Dict[str, float]):
        """
        obis_map   : from config file — field name → OBIS code
        thresholds : from config file — threshold values
        """
        # invert the map — OBIS code → field name
        # so lookup is O(1) per reading
        self.code_to_field = {v: k for k, v in obis_map.items()}
        self.thresholds    = thresholds

    def resolve_readings(
        self, readings: List[OBISReading]
    ) -> Dict[str, float]:
        """
        Convert a list of OBISReading objects to a clean dict.
        Unknown OBIS codes are ignored silently.

        Returns dict with any of these keys present:
          totalActiveEnergy, rate1Energy, rate2Energy,
          rate3Energy, rate4Energy, peakDemand,
          apparentEnergy, powerFactor, powerFactorImport,
          reactiveQI, reactiveQII, reactiveQIII, reactiveQIV
        """
        result = {}
        for r in readings:
            field = self.code_to_field.get(r.obisCode)
            if field:
                result[field] = float(r.value)
        return result

    def get_field(
        self,
        resolved: Dict[str, float],
        field_name: str,
        default: float = 0.0,
    ) -> float:
        """Get a field value with a default if missing."""
        return resolved.get(field_name, default)

    def extract_current(
        self, readings: List[OBISReading]
    ) -> Dict[str, float]:
        """
        Extract all current month meter values into clean named fields.
        Returns a standardised dict regardless of which OBIS codes
        the utility uses.
        """
        r = self.resolve_readings(readings)

        return {
            "activeEnergy":    self.get_field(r, "totalActiveEnergy", 0.0),
            "rate1":           self.get_field(r, "rate1Energy",       0.0),
            "rate2":           self.get_field(r, "rate2Energy",       0.0),
            "rate3":           self.get_field(r, "rate3Energy",       0.0),
            "rate4":           self.get_field(r, "rate4Energy",       0.0),
            "peakDemand":      self.get_field(r, "peakDemand",        0.0),
            "apparentEnergy":  self.get_field(r, "apparentEnergy",    0.0),
            "powerFactor":     self.get_field(r, "powerFactor",       0.95),
            "powerFactorImp":  self.get_field(r, "powerFactorImport", 0.95),
            "reactiveQI":      self.get_field(r, "reactiveQI",        0.0),
            "reactiveQII":     self.get_field(r, "reactiveQII",       0.0),
            "reactiveQIII":    self.get_field(r, "reactiveQIII",      0.0),
            "reactiveQIV":     self.get_field(r, "reactiveQIV",       0.0),
        }

    def extract_history_energy(
        self, history: List[HistoricalMonth]
    ) -> List[float]:
        """
        Extract ordered list of monthly active energy values from history.
        Sorted oldest first — same order as training data.
        Returns list of floats in Wh.
        """
        monthly = []
        for month in sorted(history, key=lambda m: m.billingPeriod):
            r = self.resolve_readings(month.readings)
            energy = self.get_field(r, "totalActiveEnergy", 0.0)
            monthly.append(max(energy, 0.0))
        return monthly

    def extract_history_pf(
        self, history: List[HistoricalMonth]
    ) -> List[float]:
        """
        Extract ordered list of monthly power factor values from history.
        Sorted oldest first.
        """
        monthly = []
        for month in sorted(history, key=lambda m: m.billingPeriod):
            r  = self.resolve_readings(month.readings)
            pf = self.get_field(r, "powerFactor", 0.95)
            monthly.append(float(max(min(pf, 1.0), 0.01)))
        return monthly

    def extract_history_peak(
        self, history: List[HistoricalMonth]
    ) -> List[float]:
        """
        Extract ordered list of monthly peak demand values from history.
        Sorted oldest first.
        """
        monthly = []
        for month in sorted(history, key=lambda m: m.billingPeriod):
            r    = self.resolve_readings(month.readings)
            peak = self.get_field(r, "peakDemand", 0.0)
            monthly.append(max(peak, 0.0))
        return monthly

    def get_threshold(
        self, key: str, default: float = 0.0
    ) -> float:
        """Get a threshold value from config."""
        return float(self.thresholds.get(key, default))


def build_resolver(
    obis_configs: Dict[str, Any],
    utility_id:   str,
) -> OBISResolver:
    """
    Factory function — loads the right config for the utility
    and returns a ready-to-use resolver.

    Falls back to 'default' config if utility-specific one not found.
    """
    cfg = obis_configs.get(utility_id) or obis_configs.get("default") or {}

    obis_map   = cfg.get("obisMapping",  {})
    thresholds = cfg.get("thresholds",   {})

    if not obis_map:
        raise ValueError(
            f"No OBIS mapping found for utilityId '{utility_id}'. "
            f"Available configs: {list(obis_configs.keys())}"
        )

    return OBISResolver(obis_map, thresholds)