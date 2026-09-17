# HTAD

**HTAD: Domain-Conditioned Dynamic Adversarial Learning for Heterogeneous Multi-Domain Time Series Anomaly Detection**

HTAD is a heterogeneous multi-domain multivariate time-series anomaly detection framework. It combines domain-specific representation learning, language-derived domain conditioning, causal-global temporal reconstruction, and reliability-aware adversarial learning.

## Overview

HTAD is designed for joint anomaly detection across heterogeneous multivariate time-series domains with different variable dimensions and system characteristics.

The framework contains four main components:

- **Domain-specific representation learning**  
  Lightweight domain-specific autoencoders map heterogeneous temporal patches into a common-dimensional latent interface.

- **Domain-conditioned modeling**  
  Predefined textual domain descriptions are represented using frozen pretrained GPT-2 token embeddings and used as explicit domain-level conditional information.

- **Causal-global reconstruction**  
  A shared temporal reconstruction network combines causally constrained self-attention with cross-attention over causally accessible encoder states.

- **Reliability-aware adversarial learning**  
  Reconstruction-based reliability estimation is used to reduce the influence of poorly reconstructed training samples during adversarial optimization.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
