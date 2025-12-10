# 🌐 Urban Street Network Analysis Pipeline  
*Object-Oriented Geospatial & Graph-Theoretic Framework for Academic Research*

This repository contains the complete, object-oriented Python pipeline used for the geographic and graph-theoretic analysis presented in the paper:

**Urban Street Network Configuration and Property Crime**

The pipeline preprocesses large geospatial street-centerline datasets and computes a comprehensive suite of network measures—including centrality, density, and shortest-path metrics—across multiple spatial scales for predefined study areas (e.g., Neighborhood Profile Areas / NPAs).

---

## 🚀 Key Features

### ✔ Object-Oriented Architecture  
All analysis logic is encapsulated in the `StreetNetworkProcessor` class for modularity, reusability, and extensibility.

### ✔ Automated Network Simplification  
Removes redundant degree-2 nodes to create a topologically clean network suitable for network science and spatial analysis.

### ✔ Multi-Scale Buffer Analysis  
Supports calculation of metrics at configurable spatial scales (e.g., `0.25`, `0.5`, `1.0` miles), enabling sensitivity checks and robustness analysis.

---

## 🛠️ Setup and Installation

### 1. Install Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

## 📁 Data Preparation

The input datasets should follow the paths defined in src/run_analysis.py.

| Variable            | Description                                      
| ------------------- | ----------------------------------------------- 
| `source_edges_path` | Study area street centerline dataset (GeoDataFrame) `../data/streetnetwork` 
| `NPA_shape_path`    | Study area neighborhood polygons (GeoDataFrame) `../data/neighborhoods` 


Output files
The pipeline generates:

NPA_*.shp (one per study area) in `../data/neighborhoods`

Final_dataset.shp (merged dataset) in `../data/neighborhoods`


## ▶️ Running the Analysis

The main execution script is:

```bash
python src/run_analysis.py
```
## Optional Configuration

You can modify the analysis scales in `src/run_analysis.py:`

```bash
buffer_list_mile = [0.25, 0.5, 1.0]
```
When running, the console will display progress bars for:

* Node extraction

* Network simplification

* Each neighborhood’s analysis


## 📂 Repository Structure

```
├── data/                       # Input geospatial datasets (shapefiles)
│   ├── neighborhoods/
│   └── streetnetwork/
├── src/
│   ├── run_analysis.py         # Main execution script
│   └── street_network_processor.py
├── requirements.txt
└── README.md
```

## 📜 License

MIT License

Copyright (c) 2025 Erfan Kefayat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## 🙌 Citation

If you use this code in your research, please cite the associated paper:

```
Kefayat, E., & Thill, J.-C. (2025). Urban Street Network Configuration and Property Crime: An Empirical Multivariate Case Study. ISPRS International Journal of Geo-Information, 14(5), 200.
https://doi.org/10.3390/ijgi14050200
```

## 💬 Contact

For questions or collaboration, feel free to open an issue or email me at:

```
ekefayat@charlotte.edu
```

