# 🚗 Spatiotemporal Traffic Forecasting with Graph Neural Networks

**Comparing graph adjacency structures for DCRNN-based traffic speed prediction on the METR-LA dataset**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Traffic congestion costs the US economy over **$87 billion annually**. Accurate short-term traffic forecasting is critical for navigation systems, urban planning, and autonomous vehicles. This project implements and compares **five different graph adjacency structures** for the [Diffusion Convolutional Recurrent Neural Network (DCRNN)](https://arxiv.org/abs/1707.01926), a state-of-the-art spatiotemporal model for traffic forecasting.

The key insight: **how you define "neighbor" between traffic sensors dramatically impacts prediction quality.**

---

## 🧠 Problem Statement

Given historical traffic speed readings from 207 sensors across Los Angeles County (METR-LA dataset), predict traffic speeds 15 minutes into the future. The challenge is determining the optimal graph structure that captures spatial dependencies between sensors for the DCRNN's diffusion convolution operation.

---

## 📊 Dataset

- **METR-LA**: 207 traffic sensors on LA County highways
- **Time Range**: March–June 2012
- **Granularity**: 5-minute intervals (~34,000 timesteps)
- **Features**: Traffic speed (mph) per sensor
- **Split**: 70% train / 15% validation / 15% test

---

## 🔬 Approach

### Five Graph Adjacency Structures

| Graph Type | Description | Construction Method |
|---|---|---|
| **Road Network** | Real driving distances between sensors | OSMnx + Dijkstra shortest paths + Gaussian kernel |
| **Euclidean** | Straight-line distance between sensors | Pairwise lat/lon distance + Gaussian kernel |
| **Correlation-Based** | Dynamic per-batch Pearson correlation | Computed on each training batch (12 timesteps) |
| **Fully Connected** | Every sensor connected to every other | Uniform 1/N weights |
| **Identity** | No spatial connections (baseline) | Diagonal matrix |

### Key Technical Contributions

1. **Row-normalization of adjacency matrices** — Discovered that unnormalized matrices cause the identity graph to artificially outperform spatial graphs due to diffusion convolution amplifying features at inconsistent scales. Row-normalizing (making matrices row-stochastic) resolved this and produced the expected ranking.

2. **Gaussian kernel thresholding** — Applied distance-based thresholds (0.1° for Euclidean, 10 km for road network) to create sparse adjacency matrices, improving both computational efficiency and model performance.

3. **Naive baseline comparison** — Implemented a "last observed value" predictor to validate that all DCRNN models learn beyond simple temporal persistence.

---

## 📈 Results

| Graph Type | Test MSE | Test MAE | vs. Naive Baseline |
|---|---|---|---|
| **Road Network** | **0.0071** | **0.0440** | ↓ 17.4% MSE |
| Euclidean | 0.0072 | 0.0454 | ↓ 16.3% MSE |
| Correlation-Based | 0.0072 | 0.0440 | ↓ 16.3% MSE |
| Fully Connected | 0.0073 | 0.0459 | ↓ 15.1% MSE |
| Identity | 0.0075 | 0.0456 | ↓ 12.8% MSE |
| Naive Baseline | 0.0086 | 0.0461 | — |

**Road network adjacency achieves the best performance**, confirming that real driving connectivity captures traffic flow dynamics better than geometric proximity or statistical correlation.

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch (DCRNN with diffusion convolution)
- **Graph Construction**: OSMnx, NetworkX, SciPy
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Visualization**: Matplotlib
- **Infrastructure**: Google Colab (G4 GPU)
- **Spatial Data**: OpenStreetMap (via Overpass API)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch osmnx networkx pandas numpy scikit-learn matplotlib
```

### Run
1. Open `HW3_VaradTawde.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → G4 GPU)
3. Run all cells sequentially
4. Road network computation takes ~20 min (cached after first run)

---

## 📁 Project Structure

```
├── HW3_VaradTawde.ipynb        # Main notebook with all code and analysis
├── metr-la.csv                  # METR-LA traffic speed data
├── graph_sensor_locations.csv   # Sensor coordinates (lat/lon)
├── road_network_adj.npy         # Cached road network adjacency matrix
├── graph_comparison.png         # Results visualization
└── README.md
```

---

## 💡 Key Findings

1. **Graph structure matters** — Road network adjacency outperforms identity by 17.4% in MSE, proving spatial information improves traffic forecasting
2. **Normalization is non-negotiable** — Without row-normalizing adjacency matrices, results are misleading and incomparable across graph types
3. **Correlation captures complementary patterns** — Dynamic correlation-based graphs match Euclidean performance without requiring geographic data
4. **Short horizons compress differences** — At 15-minute horizons, temporal autocorrelation dominates; spatial graphs would likely show larger gains at 1–2 hour horizons

---

## 🔮 Future Extensions

- **Longer forecast horizons** (1–2 hours) where spatial dependencies become more critical
- **Attention-based adjacency** learning (Graph Attention Networks) to let the model discover optimal graph structure
- **Multi-scale graphs** combining road network + correlation for hybrid spatial-functional connectivity
- **Real-time inference** pipeline with streaming sensor data
- **Transfer learning** to other cities' traffic networks

---

## 📚 References

- Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926). ICLR 2018.
- METR-LA Dataset: Los Angeles County traffic sensor network

---

## 👤 Author

**Varad Tawde**
- MS in Computer Science, USC
- [LinkedIn](https://linkedin.com/in/your-profile)

---

*Built as part of CSCI 587: Spatial Databases at USC*
