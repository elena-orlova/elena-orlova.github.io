---
title: "Training neural operators to preserve invariant measures of chaotic attractors"
collection: publications
#permalink: /publication/2010-10-01-paper-title-number-2
excerpt: 'We introduce a framework preserving invariant measures of chaotic attractors, offering two approaches: one utilizing optimal transport distance with expert knowledge about the system dynamics and another using a contrastive learning framework without specialized information. Empirical results demonstrate the effectiveness of our method in preserving invariant measures across various chaotic systems.'
date: 2023-11-12
venue: 'NeurIPS 2023'
#paperurl: 'http://academicpages.github.io/files/paper2.pdf'
citation: 'R. Jiang, PY Lu, E. Orlova, R. Willett'
---

Chaotic systems make long-horizon forecasts difficult because small perturbations in initial conditions cause trajectories to diverge at an exponential rate. In this setting, neural operators trained to minimize squared error losses, while capable of accurate short-term forecasts, often fail to reproduce statistical or structural properties of the dynamics over longer time horizons and can yield degenerate results. In this paper, we propose an alternative framework designed to preserve invariant measures of chaotic attractors that characterize the time-invariant statis- tical properties of the dynamics. Specifically, in the multi-environment setting (where each sample trajectory is governed by slightly different dynamics), we consider two novel approaches to training with noisy data. First, we propose a loss based on the optimal transport distance between the observed dynamics and the neural operator outputs. This approach requires expert knowledge of the underlying physics to determine what statistical features should be included in the optimal transport loss. Second, we show that a contrastive learning framework, which does not require any specialized prior knowledge, can preserve statistical properties of the dynamics nearly as well as the optimal transport approach. On a variety of chaotic systems, our method is shown empirically to preserve invariant measures of chaotic attractors.

[NeurIPS page](https://neurips.cc/virtual/2023/poster/72621) [Paper](https://arxiv.org/abs/2306.01187) [Code]( https://github.com/roxie62/neural_operators_for_chaos)