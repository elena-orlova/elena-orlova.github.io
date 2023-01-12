---
title: "Tensorized embedding layers"
collection: publications
#permalink: /publication/2010-10-01-paper-title-number-2
excerpt: 'We introduce a novel way of parameterizing embedding layers based on the Tensor Train decomposition, which allows compressing the model significantly at the cost of a negligible drop or even a slight gain in performance. We evaluate our method on a wide range of benchmarks in natural language processing and analyze the trade-off between performance and compression ratios for a wide range of architectures, from MLPs to LSTMs and Transformers.'
date: 2020-10-01
venue: 'EMNLP 2020'
#paperurl: 'http://academicpages.github.io/files/paper2.pdf'
citation: 'O. Hrinchuk, V. Khrulkov, L. Mirvakhabova, E. Orlova, I. Oseledets'
---

The embedding layers transforming input words into real vectors are the key components of deep neural networks used in natural language processing. However, when the vocabulary is large, the corresponding weight matrices can be enormous, which precludes their deployment in a limited resource setting. We introduce a novel way of parameterizing embedding layers based on the Tensor Train decomposition, which allows compressing the model significantly at the cost of a negligible drop or even a slight gain in performance. We evaluate our method on a wide range of benchmarks in natural language processing and analyze the trade-off between performance and compression ratios for a wide range of architectures, from MLPs to LSTMs and Transformers.

[Download paper here](https://aclanthology.org/2020.findings-emnlp.436/) [Code](https://github.com/KhrulkovV/tt-pytorch)