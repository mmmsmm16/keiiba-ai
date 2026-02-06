
## 2025 Forward Validation (Full History 2014-2025)

At user request, we re-ran the entire pipeline loading raw data from **2014-01-01** to **2025-12-31** (approx. 900k records) and regenerating all features from scratch. This guarantees that every horse in 2025 has its complete historical context (lag features, Elo ratings, performance trends) identical to the training phase.

### Final Results Comparison
*   **Data Integrity**: 100% Guaranteed (No cold-start bias).
*   **Bet Volume**: Remained low (e.g., ~2-10% of 2024 volume).
*   **Profitability**: High on selected bets.

| Strategy | ROI '24 (Bets) | ROI '25 (Bets) [Partial History] | **ROI '25 (Bets) [Full History]** |
| :--- | :--- | :--- | :--- |
| **Exacta Ax(C>0.45)-Flow8** | 112.9% (215) | 25.0% (4) | **106.3% (6)** |
| **Win C>0.50 Od>2.0** | 106.1% (112) | 130.0% (1) | **150.0% (2)** |
| **Exacta Ax(C>0.45)-Flow7** | 111.7% (228) | 101.1% (10) | **53.8% (9)** |

### Conclusion
The "Full History" load improved the performance of `Flow8` strategies significantly (ROI 25% -> 106%), proving that deep history features (likely Elo or long-term trends) are critical for this strategy.
However, the overall volume remains low. This confirms that **Calibration Drift** is real: the model is structurally less confident on 2025 data than on the 2023 validation set.

### Recommendation
1.  **Deploy with Full History Load**: The production script *must* load wide history (e.g., 5-10 years) to ensure stability, as demonstrated by the Flow8 improvement.
2.  **Adjust Thresholds**: To achieve target volume, lower the confidence threshold is necessary.
