# FireRhythm — Systems Engineering Pack

## 1. Stakeholders & Concept of Operations

| Stakeholder | Need | How FireRhythm serves it |
| --- | --- | --- |
| Provincial / state agriculture departments | Know where and when crop-residue burning will peak | Burn calendar, onset prediction, deployment priority list |
| Environmental protection agencies | Target enforcement and smog warnings | Risk map, anomaly flags, live detections |
| Disaster management authorities & firefighters | Spot unusual fire activity early | High/Medium/Low risk with explanations, live FIRMS feed |
| Researchers | A consistent 2000–present fire record | Harmonized MODIS + VIIRS series, CSV export |

**ConOps scenario.** On 1 October, an officer at the Punjab agriculture department opens FireRhythm and draws the
district boundary. The burn calendar shows that October–November carries most of the year's burning. The risk tab
flags cells where onset is running early and the forecast exceeds the usual November level. The officer downloads
the deployment list and sends residue-management machinery (super seeders, balers) to the top-ranked cells first.
Throughout the season, the live map confirms where burning is actually happening.

## 2. Functional Architecture

```mermaid
flowchart LR
  A[NASA FIRMS API<br/>MODIS C6.1 + VIIRS S-NPP] -->|I-1 CSV| S1[S1 Ingestion<br/>clean, filter, 0.25° grid]
  S1 -->|I-2 Parquet| S2[S2 Harmonization<br/>LightGBM transfer function]
  S2 -->|I-3 Parquet| S3[S3 Signal Analytics<br/>STL, FFT, regimes, onset]
  S3 -->|I-4 Parquet| S4[S4 Risk Engine<br/>anomaly, drift, forecast, SHAP]
  S4 -->|I-5 Parquet/CSV| S5[S5 Web UI<br/>Streamlit + pydeck]
  A -.->|I-6 live NRT| S5
```

## 3. Interface Control Table

| ID | From → To | Data | Format | Update rate |
| --- | --- | --- | --- | --- |
| I-1 | FIRMS → S1 | Hotspot points (lat, lon, date, time, FRP, confidence) | CSV via REST API | Archive once; NRT within hours |
| I-2 | S1 → S2 | Cell × month counts per sensor (zeros filled) | Parquet | Monthly |
| I-3 | S2 → S3 | Harmonized VIIRS-equivalent counts + source flag | Parquet | Monthly |
| I-4 | S3 → S4 | Fingerprints, regimes, onset timing, STL components | Parquet | Monthly |
| I-5 | S4 → S5 | Risk score, explanation, forecast, deployment rank | Parquet / CSV | Monthly |
| I-6 | FIRMS → S5 | Last 1–5 days of NRT detections for the selected area | CSV via REST API | On demand, cached 15 min |

## 4. Trade Studies (weights sum to 1; scores 1–5, team assessment)

**4a. Harmonization method**

| Criterion (weight) | Fixed ratio | Linear regression | LightGBM |
| --- | --- | --- | --- |
| Accuracy on held-out years (0.4) | 2 | 3 | 5 |
| Robustness to sensor drift (0.2) | 1 | 2 | 4 |
| Explainability (0.2) | 5 | 4 | 3 |
| Build effort (0.2) | 5 | 4 | 3 |
| **Weighted total** | **2.6** | **3.2** | **4.0 ✅** |

Evidence: on 2023+ test years, LightGBM cut cell RMSE by 23% versus the fixed ratio. The ratio's R² fell from
0.84 (2021+) to 0.54 (2023+) as the MODIS orbits drifted.

**4b. Grid resolution**

| Criterion (weight) | 0.1° | 0.25° | 0.5° |
| --- | --- | --- | --- |
| Spatial detail for field teams (0.35) | 5 | 4 | 2 |
| Statistical stability (counts per cell) (0.35) | 2 | 4 | 5 |
| App speed (0.3) | 2 | 4 | 5 |
| **Weighted total** | **3.1** | **4.0 ✅** | **3.9** |

## 5. Requirements Traceability & Verification Matrix

| ID | Requirement | Subsystem | Method | Result | Status |
| --- | --- | --- | --- | --- | --- |
| FR-01 | Select area by drawing or lat/lon | S5 | Demo | Rectangle/polygon draw + coordinate entry working | Pass |
| FR-02 | Harmonized monthly calendar 2000–now | S2, S5 | Inspection | Year × month calendar, modeled years marked | Pass |
| FR-03 | Current month vs historical median | S5 | Test | Bar chart with median + IQR band, % difference | Pass |
| FR-04 | Flag anomalous burning (High/Med/Low) | S4 | Back-test (as of 2025-10-01) | Forecast MAE 6.00 vs 7.10 seasonal-naive (−16%); High precision 55%, recall 3%; High+Med recall 21% | Partial |
| FR-05 | Rank crop cells by predicted onset | S3, S4 | Back-test, 74 cells | Onset error 12.1 days vs 12.9 climatology | Pass (small gain) |
| FR-06 | CSV download of hotspots | S5 | Test | Hotspot + deployment CSV downloads | Pass |
| PR-01 | No structural break at 2012 sensor switch | S2 | Change-point + t-test | Naive p = 3.3e-11 → harmonized p = 0.069; mild shift near 2011-09 | Pass |
| PR-02 | Beat ratio baseline on held-out years | S2 | Back-test 2023+ | Cell RMSE 20.55 → 15.83; regional R² 0.68 → 0.83 | Pass |
| PR-03 | Area query renders < 3 s | S5 | Timing | __ s (fill in from the sidebar) | __ |
| NFR-01 | Colorblind-safe colour ramps | S5 | Inspection | Inferno / viridis (perceptually uniform) throughout | Pass |

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| FIRMS API limits or outage | Medium | High | Archive cached as Parquet; app reads files; live mode degrades gracefully |
| Pre-2012 harmonization error (Terra-only 2000–02, residual 2011 shift) | Medium | Medium | Modeled years labelled; validated with change-point test |
| MODIS & S-NPP orbit drift / end of life | High | Medium | Learned transfer function; live mode already uses NOAA-20/21 VIIRS |
| Static / industrial heat sources in NRT (no type flag) | Medium | Medium | Deployment list requires ≥ 50 hotspots/yr burning history |
| Risk score lacks weather inputs | High | Medium | Labelled "anomalous burning risk"; ERA5 dryness on roadmap |
| Low early-season recall of High-risk flag | High | Medium | Report honestly; re-score as in-season data arrives |

## 7. Lifecycle (V-model)
Stakeholder needs → requirements (§5) → architecture (§2) → subsystems S1–S5 → subsystem verification
(PR-01, PR-02 back-tests) → integration (app) → system verification (§5 results) → validation (ConOps demo).