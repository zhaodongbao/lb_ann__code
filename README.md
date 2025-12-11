# Efficient Acceleration of Multivariate DTW Distance (EMA-DTW)

A high-performance algorithm for accelerating multivariate Dynamic Time Warping (DTW) distance computation in large-scale sequence data.

## Overview

This repository contains the official implementation of the **LB_ANN** lower-bound function and associated acceleration strategies for multivariate DTW distance calculation. The proposed method leverages approximate nearest neighbor (ANN) search techniques to provide significantly tighter bounds compared to existing approaches, enabling efficient k-nearest neighbor (KNN) similarity queries and threshold-based similarity queries for multivariate sequences.

## Features

* **Novel Lower-Bound Function**: Implements LB_ANN, a theoretically proven tighter lower bound for multivariate DTW distance
* **ANN-Based Acceleration**: Utilizes Annoy trees for efficient high-dimensional vector similarity search
* **Multiple Acceleration Strategies**: 
  - Early termination based on lower-bound thresholds
  - Trial computation with restricted windows
  - Advanced pruning techniques in the cost matrix
* **Robust Performance**: Maintains acceleration effectiveness even without window constraints
* **Comprehensive Evaluation**: Validated on six representative multivariate datasets from UEA archive

## Installation & Usage

### Prerequisites

* Python 3.8+
* NumPy, SciPy
* Annoy (spotify/annoy)
* (Optional) UEA multivariate time series datasets for benchmarking

### Installation
git clone https://github.com/your-username/EMA-DTW.git
cd EMA-DTW
pip install -r requirements.txt

