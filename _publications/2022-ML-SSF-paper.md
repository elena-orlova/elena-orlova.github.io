---
title: "Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting"
collection: publications
#permalink: /publication/2009-10-01-paper-title-number-1
# excerpt: 'Operational forecasting has long lacked accurate temperature and precipitation predictions on subseasonal time scales - two to two months in advance. These forecasts would have immense value in agriculture, insurance, and economics. Our paper describes an application of machine learning techniques to forecast monthly average precipitation and 2-meter temperature using physics-based predictions and observational data two weeks in advance for the entire continental United States. The proposed models outperform common benchmarks such as historical averages and averages of physics-based predictors. Our fundings suggest that utilizing the full set of physics-based predictions instead of the average enhances the accuracy of the final forecast.'
excerpt: ''
date: 2022-09-12
venue: 'Artificial Intelligence for the Earth Systems (AIES)'
#paperurl: 'http://elena-orlova.github.io/files/paper_SSF.pdf'
citation: 'E. Orlova, H. Liu, R. Rossellini, B. Cash, R. Willett'
---

[Journal paper](https://doi.org/10.1175/AIES-D-23-0103.1)

Producing high-quality forecasts of key climate variables, such as temperature and precipitation, on subseasonal time scales has long been a gap in operational forecasting. This study explores an application of machine learning (ML) models as post-processing tools for subseasonal forecasting. Lagged numerical ensemble forecasts (i.e., an ensemble where the members have different initialization dates) and observational data, including relative humidity, pressure at sea level, and geopotential height, are incorporated into various ML methods to predict monthly average precipitation and two-meter temperature two weeks in advance for the continental United States. For regression, quantile regression, and tercile classification tasks, we consider using linear models, random forests, convolutional neural networks, and stacked models (a multi-model approach based on the prediction of the individual ML models). Unlike previous ML approaches that often use ensemble mean alone, we leverage information embedded in the ensemble forecasts to enhance prediction accuracy. Additionally, we investigate extreme event predictions that are crucial for planning and mitigation efforts. Considering ensemble members as a collection of spatial forecasts, we explore different approaches to using spatial information. Trade-offs between different approaches may be mitigated with model stacking. Our proposed models outperform standard baselines such as climatological forecasts and ensemble means. In addition, we investigate feature importance, trade-offs between using the full ensemble or only the ensemble mean, and different modes of accounting for spatial variability.


[Paper](https://arxiv.org/abs/2211.15856) [Code and data](https://github.com/elena-orlova/SSF-project)

<!-- Recommended citation: Your Name, You. (2009). "Paper Title Number 1." <i>Journal 1</i>. 1(1). -->