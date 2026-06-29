"""
Response Models
===============
Defines exactly what the engine sends back to HES
after validating a meter reading.

Every field is meaningful and actionable for HES.
HES uses this to decide whether to alert, review or monitor.
"""

from pydantic import BaseModel
from typing import Optional


class ModelOutputs(BaseModel):
    """
    Raw outputs from both ML models.
    Included so HES or analysts can see exactly what each model said.
    """
    # Isolation Forest
    ifIsAnomaly:    bool
    ifScore:        float   # lower = more anomalous, range -1.0 to 0.0

    # TFT
    tftPredictedWh: float   # p50 median forecast
    tftP10:         float   # lower bound — actual below this = SuddenDrop
    tftP90:         float   # upper bound — actual above this = SuddenSpike
    tftDeviation:   float   # (actual - predicted) / predicted × 100
    tftSuddenDrop:  bool    # actual < p10
    tftSuddenSpike: bool    # actual > p90


class RuleFlags(BaseModel):
    """
    Boolean flags from the validation rules engine.
    Each flag corresponds to one specific billing check.
    """
    billingMismatch:        bool = False  # billed != metered
    rateSumError:           bool = False  # Rate1+2+3+4 != Total
    zeroConsumption:        bool = False  # active energy = 0
    flatLine:               bool = False  # same value 3+ months
    powerFactorLow:         bool = False  # PF < 0.85
    powerFactorDeviation:   bool = False  # PF dropped > 8% from avg
    ntlSuspected:           bool = False  # drop + PF degradation
    bypassSuspected:        bool = False  # drop but peak demand high
    tariffBoundaryGaming:   bool = False  # reading near slab boundary


class ValidationResponse(BaseModel):
    """
    Complete validation result returned to HES.

    Key fields HES acts on:
      severity      → HIGH triggers immediate alert
      ntlSuspected  → flags for field inspection
      preInvoiceFlag→ holds billing until reviewed
      recommendation→ tells HES what action to take
    """
    # identity
    meterId:          str
    billingPeriod:    str
    processedAt:      str   # ISO timestamp

    # risk assessment
    riskScore:        float          # 0 to 100
    severity:         str            # HIGH / MEDIUM / LOW
    isAnomaly:        bool
    anomalyType:      Optional[str]  # what type of anomaly

    # consumption comparison
    actualWh:         float   # what meter recorded
    predictedWh:      float   # what TFT expected
    deviationPct:     float   # % difference
    historicalAvgWh:  float   # this meter's historical average

    # billing
    billedAmountWh:   float   # what billing system charged
    billAmountINR:    float   # rupee amount
    revenueAtRiskINR: float   # estimated revenue at risk

    # flags
    ntlSuspected:     bool    # non-technical loss suspected
    preInvoiceFlag:   bool    # hold billing pending review

    # model confidence
    historyMonths:    int     # how many months of history available
    modelConfidence:  str     # HIGH / MEDIUM / LOW / INSUFFICIENT

    # model details
    modelOutputs:     ModelOutputs
    ruleFlags:        RuleFlags

    # human readable
    explanation:      str     # plain English description
    recommendation:   str     # FIELD_INSPECTION / REVIEW / MONITOR

    class Config:
        json_schema_extra = {
            "example": {
                "meterId":         "A3260060",
                "billingPeriod":   "2026-03",
                "processedAt":     "2026-03-01T09:14:07Z",
                "riskScore":       100.0,
                "severity":        "HIGH",
                "isAnomaly":       True,
                "anomalyType":     "NTL_Suspected",
                "actualWh":        1110.0,
                "predictedWh":     15820.0,
                "deviationPct":    -92.99,
                "historicalAvgWh": 14600.0,
                "billedAmountWh":  1110.0,
                "billAmountINR":   8.88,
                "revenueAtRiskINR":116.96,
                "ntlSuspected":    True,
                "preInvoiceFlag":  True,
                "historyMonths":   11,
                "modelConfidence": "HIGH",
                "modelOutputs": {
                    "ifIsAnomaly":    True,
                    "ifScore":        -0.69,
                    "tftPredictedWh": 15820.0,
                    "tftP10":         14200.0,
                    "tftP90":         17400.0,
                    "tftDeviation":   -92.99,
                    "tftSuddenDrop":  True,
                    "tftSuddenSpike": False
                },
                "ruleFlags": {
                    "billingMismatch":      False,
                    "rateSumError":         False,
                    "zeroConsumption":      False,
                    "flatLine":             False,
                    "powerFactorLow":       True,
                    "powerFactorDeviation": True,
                    "ntlSuspected":         True,
                    "bypassSuspected":      False,
                    "tariffBoundaryGaming": False
                },
                "explanation":   "Meter A3260060 recorded 1,110 Wh in March 2026, a 93% drop from its historical average of 14,600 Wh. Power factor also degraded from 0.99 to 0.83. This pattern is consistent with meter tampering or bypass. Immediate field inspection recommended. Revenue at risk: Rs 116.96.",
                "recommendation":"FIELD_INSPECTION"
            }
        }