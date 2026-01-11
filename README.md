**Histogram Confidence Method (HCM)**

A lightweight distribution monitoring primitive for real-time and embedded systems

The Histogram Confidence Method (HCM) is a simple, interpretable method for monitoring distributional drift in streaming data. HCM is designed as a distribution monitoring primitive, not a predictive model or change-point estimator.

**Key properties**

-No predictive model

-No retraining

-No online statistical estimation

-Fixed computational cost per update

-Suitable for microcontrollers, edge devices, and long-running systems

HCM constructs a fixed baseline histogram from representative data using adaptive binning to ensure sufficient statistical mass per bin. During operation, incoming observations are aggregated in fixed-size windows and compared against precomputed confidence bounds. Sustained deviations from the baseline envelope indicate drift.

This repository contains a compact reference implementation (hcm.py) suitable for:

-sensor and signal drift monitoring

-industrial and process monitoring

-ML input distribution monitoring

-anomaly and deviation detection

-real-time and embedded systems

**Reference**

Whitepaper:
Steven F. Srebranig (2026). The Histogram Confidence Method (HCM): A Lightweight Distribution Monitoring Primitive for Drift Detection.
Zenodo DOI: 10.5281/zenodo.18204048

(If you intend to keep a separate code DOI, we should explicitly label it here.)

License

MIT License — free for all use.
