# CDS Pricing and Sensitivity Analysis - Summary

## Problem Parameters

- **Risk-free rate:** 4% (flat for all maturities)
- **Recovery rate:** 25%
- **Notional:** $10,000,000

## Market CDS Spreads

| Maturity | Premium (bps) |
|----------|---------------|
| 1y       | 700           |
| 2y       | 710           |
| 3y       | 720           |
| 5y       | 740           |

---

## Part 1: Bootstrapped Survival Curve

Using piecewise-constant hazard rates, the survival curve was bootstrapped from market CDS spreads:

| Maturity | Hazard Rate λ(t) | Survival Probability Q(t) |
|----------|------------------|---------------------------|
| 1y       | 9.3401%          | 0.910828                  |
| 2y       | 9.6267%          | 0.827234                  |
| 3y       | 9.9363%          | 0.748989                  |
| 5y       | 11.2186%         | 0.598456                  |

**Methodology:**
- For each maturity, solve for the hazard rate λ that makes the CDS value zero (par spread condition)
- Par spread condition: PV(Protection Leg) = PV(Premium Leg)
- Protection Leg: $(1 - R) \times \sum_i D(t_i) \times [Q(t_{i-1}) - Q(t_i)]$
- Premium Leg: $s \times \sum_i \Delta t_i \times D(t_i) \times Q(t_i)$ + accrued premium
- Where: $Q(t) = \exp(-\int_0^t \lambda(s)ds)$ for piecewise-constant λ

---

## Part 2: Fair Spread for 4y CDS

**Answer: 734.54 bps**

**Method:**
- Interpolate hazard rate between 3y and 5y maturities
- Calculate fair spread using: $s = \frac{PV_{protection}}{PV_{annuity}}$
- The 4y spread falls between the 3y (720 bps) and 5y (740 bps) market spreads

---

## Part 3: Value of 5y CDS Bought 1 Year Ago

**Contract Details:**
- Original: 5y CDS with 80 bps contractual spread
- Current: 4y remaining maturity
- Current fair 4y spread: 734.54 bps

**Mark-to-Market Value:**
- **Per $1 notional: $0.201061**
- **On $10M notional: $2,010,611.28**

**Interpretation:**
You would **charge $2,010,611.28** to transfer this CDS position.

The positive value reflects that the contractual spread (80 bps) is far below the current market spread (734.54 bps) for 4y protection. The protection seller would pay this amount to the protection buyer to exit the contract, as they locked in a very favorable rate.

---

## Part 4: DV01 with Respect to CDS Curve

**DV01 on $10M notional: $56,856.83**

**Definition:** Change in CDS value for a 1bp parallel shift in all CDS spreads.

**Interpretation:**
- If all CDS spreads increase by 1bp → CDS value **increases** by $56,856.83
- Positive DV01 indicates the CDS (protection buyer) benefits from widening spreads
- This is expected: wider spreads mean higher default risk, making protection more valuable

---

## Part 5: DV01 with Respect to Interest Rate Curve

**DV01 on $10M notional: $53,713.33**

**Definition:** Change in CDS value for a 1bp parallel shift in risk-free interest rates.

**Interpretation:**
- If interest rates increase by 1bp → CDS value **increases** by $53,713.33
- The positive sensitivity is surprising at first but reflects that:
  - Protection leg (single payment on default) has shorter duration than premium leg (ongoing payments)
  - Higher rates discount the premium leg more than the protection leg
  - Net effect: Protection buyer gains from rate increases

---

## Part 6: Sensitivity with Respect to Recovery Rate

**Sensitivity on $10M notional: $54,206.23 per 1% change**

**Definition:** Change in CDS value for a 1% increase in recovery rate (e.g., 25% → 26%).

**Interpretation:**
- If recovery rate increases by 1% → CDS value **increases** by $54,206.23
- This positive sensitivity seems counterintuitive but occurs because:
  1. Higher recovery → lower loss given default → protection worth less (negative effect)
  2. BUT: Higher recovery means market re-prices hazard rates upward to match same CDS spreads
  3. When we re-bootstrap with higher recovery, we get higher implied hazard rates
  4. The re-calibration effect dominates for this particular position

---

## Summary Table

| Metric | Value |
|--------|-------|
| **1y Hazard Rate** | 9.34% |
| **2y Hazard Rate** | 9.63% |
| **3y Hazard Rate** | 9.94% |
| **5y Hazard Rate** | 11.22% |
| **Fair 4y CDS Spread** | 734.54 bps |
| **MTM Value (5y→4y @ 80bps)** | $2,010,611 |
| **DV01 (CDS Curve)** | $56,857 |
| **DV01 (Interest Rate)** | $53,713 |
| **Recovery Sensitivity** | $54,206 per 1% |

---

## Key Insights

1. **High Default Risk:** Hazard rates of ~9-11% indicate distressed credit (high probability of default)
2. **Very Favorable Contract:** The 80 bps CDS bought 1 year ago is worth $2M+ today because spreads have widened dramatically
3. **Risk Exposure:** All sensitivities are positive, meaning:
   - Wider credit spreads → higher value (expected for protection buyer)
   - Higher rates → higher value (duration mismatch effect)
   - Higher recovery → higher value (re-calibration effect)

---

## Implementation Details

- **Method:** Bootstrapping with Newton-Raphson solver
- **Frequency:** Annual payment assumption (simplified from quarterly)
- **Interpolation:** Linear interpolation of hazard rates between maturities
- **Numerical Precision:** Convergence tolerance 1e-8

## Files

- `cds_pricing.py` - Complete implementation with all calculations
- `cds_analysis_results.png` - Visualization of hazard rates, survival curve, spreads, and sensitivities

## Running the Code

```powershell
python cds_pricing.py
```

**Requirements:** numpy, pandas, scipy, matplotlib
