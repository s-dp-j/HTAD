# HTAD

**HTAD: Domain-Conditioned Dynamic Adversarial Learning for Heterogeneous Multi-Domain Time Series Anomaly Detection**

HTAD is a heterogeneous multi-domain multivariate time-series anomaly detection framework. It combines domain-specific representation learning, language-derived domain conditioning, causal-global temporal reconstruction, and reliability-aware adversarial learning.

## Overview

HTAD is designed for joint anomaly detection across heterogeneous multivariate time-series domains with different variable dimensions and system characteristics.

The framework contains four main components:

- **Data preprocessing module**  
  Raw multivariate time series from each domain are partitioned into temporal patches and transformed by a domain-specific encoder into fixed-dimensional latent representations. Meanwhile, textual domain descriptions are represented using pretrained GPT-2 token embeddings, aggregated and projected into the latent feature space to provide explicit domain-conditioned information.

- **Time series reconstruction module**  
  The temporal and domain representations are fused and processed by a shared causal-global reconstruction network, which combines causally constrained temporal modeling with global context aggregation over all causally accessible patch positions. A domain-specific reconstruction head subsequently maps the reconstructed latent representation back to the original variable space.

- **Dynamic weighting discriminator module**  
  HTAD uses reconstruction error as a sample reliability signal and estimates robust domain-specific calibration statistics from training data. A warm-up and gradual reweighting mechanism then adaptively adjusts sample contributions to reconstruction and adversarial optimization, reducing the influence of poorly reconstructed observations while improving training robustness.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
