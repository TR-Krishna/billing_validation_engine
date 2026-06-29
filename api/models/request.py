"""
Request Models
==============
Defines exactly what HES sends to POST /api/validate.

HES sends data in OBIS key-value format — matching the real
billing profile structure from IS 16444 compliant meters.

Example payload:
{
    "utilityId": "default",
    "meterId": "A3260060",
    "meterType": "Commercial",
    "tariffSlab": "LT-2",
    "location": "Chennai",
    "billingPeriod": "2026-03",
    "historyMonths": 11,
    "currentReadings": [
        { "obisCode": "1.0.1.8.0.255", "value": 15820, "unit": "Wh" },
        { "obisCode": "1.0.1.6.0.255", "value": 3306,  "unit": "W"  },
        { "obisCode": "1.0.13.0.0.255","value": 0.99,  "unit": ""   },
        ...
    ],
    "history": [
        {
            "billingPeriod": "2026-02",
            "readings": [
                { "obisCode": "1.0.1.8.0.255", "value": 14200, "unit": "Wh" },
                ...
            ]
        },
        ...
    ]
}
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class OBISReading(BaseModel):
    """
    One OBIS code and its value.
    Matches the billing profile data format from EcoSEnter / HES.

    Example:
      obisCode: "1.0.1.8.0.255"   → total active energy
      value:    15820              → 15,820 Wh
      unit:     "Wh"
    """
    obisCode: str
    value:    float
    unit:     Optional[str] = ""


class HistoricalMonth(BaseModel):
    """
    One month of historical billing data.
    HES sends previous months so features can be computed in memory.
    No DB lookup needed — history is self-contained in the request.
    """
    billingPeriod: str               # e.g. "2026-02"
    readings:      List[OBISReading]


class ValidationRequest(BaseModel):
    """
    Complete payload HES sends for one meter validation.

    Fields:
      utilityId      → which OBIS config to use (default / TNEB / BESCOM)
      meterId        → unique meter serial number
      meterType      → Residential / Commercial / Industrial
      tariffSlab     → LT-1 / LT-2 / LT-3
      location       → city for seasonal context
      billingPeriod  → current month being validated e.g. "2026-03"
      historyMonths  → how many months of history are included
                       engine uses this to assess ML model confidence
      currentReadings→ this month's OBIS readings
      history        → previous months' OBIS readings (oldest first)
    """
    utilityId:       str = Field(default="default")
    meterId:         str
    meterType:       str = Field(
                         description="Residential / Commercial / Industrial")
    tariffSlab:      str = Field(
                         description="LT-1 / LT-2 / LT-3")
    location:        str = Field(default="Unknown")
    billingPeriod:   str = Field(
                         description="Current billing period e.g. 2026-03")
    historyMonths:   int = Field(default=0,
                         description="Number of history months included")
    currentReadings: List[OBISReading]
    history:         List[HistoricalMonth] = Field(default=[])

    class Config:
        json_schema_extra = {
            "example": {
                "utilityId":     "default",
                "meterId":       "A3260060",
                "meterType":     "Commercial",
                "tariffSlab":    "LT-2",
                "location":      "Chennai",
                "billingPeriod": "2026-03",
                "historyMonths": 11,
                "currentReadings": [
                    {"obisCode": "1.0.1.8.0.255", "value": 1110,  "unit": "Wh"},
                    {"obisCode": "1.0.1.8.1.255", "value": 0,     "unit": "Wh"},
                    {"obisCode": "1.0.1.8.2.255", "value": 1110,  "unit": "Wh"},
                    {"obisCode": "1.0.1.8.3.255", "value": 0,     "unit": "Wh"},
                    {"obisCode": "1.0.1.8.4.255", "value": 0,     "unit": "Wh"},
                    {"obisCode": "1.0.1.6.0.255", "value": 1482,  "unit": "W"},
                    {"obisCode": "1.0.9.8.0.255", "value": 1330,  "unit": "VAh"},
                    {"obisCode": "1.0.13.0.0.255","value": 0.83,  "unit": ""},
                    {"obisCode": "1.0.84.0.0.255","value": 1.0,   "unit": ""},
                    {"obisCode": "1.0.5.8.0.255", "value": 380,   "unit": "VArh"},
                    {"obisCode": "1.0.6.8.0.255", "value": 0,     "unit": "VArh"},
                    {"obisCode": "1.0.7.8.0.255", "value": 0,     "unit": "VArh"},
                    {"obisCode": "1.0.8.8.0.255", "value": 80,    "unit": "VArh"},
                ],
                "history": [
                    {
                        "billingPeriod": "2026-02",
                        "readings": [
                            {"obisCode":"1.0.1.8.0.255","value":15820,"unit":"Wh"},
                            {"obisCode":"1.0.13.0.0.255","value":0.99,"unit":""},
                            {"obisCode":"1.0.1.6.0.255","value":3306,"unit":"W"},
                        ]
                    }
                ]
            }
        }