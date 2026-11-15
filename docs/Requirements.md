# FireRhythm – Requirements & Verification

| ID | Requirement | Verification method | Result | Status |
| --- | --- | --- | --- | --- |
| FR-01 | User selects an area by drawing on a map or entering a lat/lon bbox | Demo | | |
| FR-02 | System shows a harmonized monthly burn calendar, 2000 to now | Inspection | | |
| FR-03 | System compares the current month against the historical median for the area | Test | | |
| FR-04 | System flags cells with anomalous burning (High/Med/Low) | Back-test on a past season | | |
| FR-05 | System ranks crop-burning cells by predicted onset in the next 30 days | Back-test, error in days | | |
| FR-06 | User downloads hotspots for the selection as CSV | Test | | |
| PR-01 | Harmonized series shows no structural break at the 2012 sensor switch | Change-point + t-test | Naive splice p = 3.3e-11 → harmonized p = 0.069 (not significant at 0.05); mild residual shift flagged near 2011-09 | Pass |
| PR-02 | Harmonization beats the ratio baseline on held-out years (2023+) | Back-test (RMSE, R²) | Cell RMSE 20.55 → 15.83 (−23%), R² 0.54 → 0.73; regional RMSE 4193 → 3057 (−27%), R² 0.68 → 0.83 | Pass |
| PR-03 | Area query renders in under 3 s | Timing test | | |
| NFR-01 | All color ramps are colorblind-safe | Simulator check | | |