---
title: "Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting"
collection: publications
#permalink: /publication/2009-10-01-paper-title-number-1
# excerpt: 'Operational forecasting has long lacked accurate temperature and precipitation predictions on subseasonal time scales - two to two months in advance. These forecasts would have immense value in agriculture, insurance, and economics. Our paper describes an application of machine learning techniques to forecast monthly average precipitation and 2-meter temperature using physics-based predictions and observational data two weeks in advance for the entire continental United States. The proposed models outperform common benchmarks such as historical averages and averages of physics-based predictors. Our fundings suggest that utilizing the full set of physics-based predictions instead of the average enhances the accuracy of the final forecast.'
excerpt: ''
date: 2022-12-12
venue: 'arxiv'
#paperurl: 'http://elena-orlova.github.io/files/paper_SSF.pdf'
citation: 'E. Orlova, H. Liu, R. Rossellini, B. Cash, R. Willett'
---

Producing high-quality forecasts of key climate variables such as temperature and precipitation on subseasonal time scales has long been a gap in operational forecasting. Recent studies have shown promising results using machine learning (ML) models to advance subseasonal forecasting (SSF), but several open questions remain. First, several past approaches use the average of an ensemble of physics-based forecasts as an input feature of these models. However, ensemble forecasts contain information that can aid prediction beyond only the ensemble mean. Second, past methods have focused on average performance, whereas forecasts of extreme events are far more important for planning and mitigation purposes. Third, climate forecasts correspond to a spatially-varying collection of forecasts, and different methods account for spatial variability in the response differently. Trade-offs between different approaches may be mitigated with model stacking. This paper describes the application of a variety of ML methods used to predict monthly average precipitation and two meter temperature using physics-based predictions (ensemble forecasts) and observational data such as relative humidity, pressure at sea level, or geopotential height, two weeks in advance for the whole continental United States. Regression, quantile regression, and tercile classification tasks using linear models, random forests, convolutional neural networks, and stacked models are considered. The proposed models outperform common baselines such as historical averages (or quantiles) and ensemble averages (or quantiles). This paper further includes an investigation of feature importance, trade-offs between using the full ensemble or only the ensemble average, and different modes of accounting for spatial variability.

[Download paper here](https://arxiv.org/abs/2211.15856) [Code and data](https://github.com/elena-orlova/SSF-project)

<!-- Recommended citation: Your Name, You. (2009). "Paper Title Number 1." <i>Journal 1</i>. 1(1). -->