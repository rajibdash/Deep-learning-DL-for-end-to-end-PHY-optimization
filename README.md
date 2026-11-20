# Deep Learning for End-to-End Physical-Layer Optimization

This project investigates deep-learning methods for wireless physical-layer (PHY)
optimization, covering both **downlink (DL)** and **uplink (UL)**. Here, DL means
downlink; deep learning is written out to avoid ambiguity.

## Goal

Develop and evaluate learned PHY components against conventional signal-processing
baselines, with emphasis on measurable gains, generalization to unseen channel
conditions, and practical inference cost. The initial scope is a configurable
multi-antenna OFDM link; modulation, antenna counts, channel models, and operating
conditions should be recorded with each experiment.

## Use cases and candidate methods

| Direction | PHY tasks to investigate | Candidate learning methods |
| --- | --- | --- |
| Downlink | Precoding/beamforming, CSI compression and feedback, and power or resource allocation | Model-driven deep unfolding for precoding; neural or autoencoder-based CSI compression; graph neural networks for multi-user allocation |
| Uplink | Pilot-aided channel estimation, multi-user MIMO detection, and power control | Deep-unfolded estimators and detectors; neural receivers; graph neural networks for coupled users |
| End-to-end | Joint transmitter/receiver design under channel and hardware constraints | Differentiable autoencoders, compared with modular learned components and conventional PHY chains |

These are candidates, not assumed winners. Start with one task per direction and
one model-driven method alongside a conventional baseline. Deep unfolding is a
particularly useful starting point because it makes an iterative signal-processing
algorithm trainable while retaining its structure and interpretability. Broader
architectures should be added only when they address a demonstrated limitation.

## Evaluation plan

1. Define a reproducible link configuration and generate channel realizations for
   training, validation, and testing. Keep test realizations and channel conditions
   separate from training.
2. Establish conventional baselines appropriate to each task (for example, linear
   precoding and MMSE detection), as well as a simple learned baseline.
3. Train and tune each learned method using fixed seeds and documented settings.
   Enforce PHY constraints such as transmit-power limits and valid constellation
   outputs.
4. Evaluate both nominal and shifted conditions, including changes in SNR, channel
   statistics, user loading, and antenna configuration where supported.
5. Report BER/BLER, achievable spectral efficiency or rate, and task-specific
   quality (such as channel-estimation error). Also report parameter count,
   inference latency, and computational cost.
6. Repeat experiments across seeds and report uncertainty; do not claim an
   improvement unless it is measured against the same channel realizations and
   constraints as the baseline.

## Suggested milestones

- **Baseline:** Specify the link and channel assumptions; reproduce conventional
  transmitter, channel, and receiver behavior.
- **UL prototype:** Evaluate channel estimation and multi-user detection, starting
  with deep-unfolded models.
- **DL prototype:** Evaluate precoding and CSI feedback, respecting power and
  feedback-rate constraints.
- **Comparison:** Add end-to-end learning only after the modular baselines work;
  compare performance, robustness, and inference cost.
- **Reproducibility:** Publish configurations, seeds, dataset-generation details,
  and evaluation scripts with each reported result.

No simulation code, datasets, or automated tests are included yet; this README
defines the research scope and evaluation criteria rather than claiming completed
results.

## Starting references

- T. O'Shea and J. Hoydis, “An Introduction to Deep Learning for the Physical
  Layer,” *IEEE Transactions on Cognitive Communications and Networking*, 2017.
  [arXiv:1702.00832](https://arxiv.org/abs/1702.00832)
- “Comprehensive Review of Deep Unfolding Techniques for Next-Generation Wireless
  Communication Systems,” 2025.
  [arXiv:2502.05952](https://arxiv.org/abs/2502.05952)
- 3GPP, [TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz](https://www.3gpp.org/DynaReport/38901.htm),
  for standardized channel-model assumptions where applicable.
