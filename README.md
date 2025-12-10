Urban Street Network Analysis Pipeline (Academic Repository)

This repository contains the complete, object-oriented Python pipeline used for the geographic and graph-theoretic analysis presented in the paper: Urban Street Network Configuration and Property Crime.

The code pre-processes large geospatial street centerline datasets and calculates a comprehensive set of network measures (centrality, density, shortest path) for defined study areas (NPAs) across multiple buffer scales.

🚀 Key Features

Object-Oriented Design: Analysis logic is encapsulated in the StreetNetworkProcessor class for modularity.

Performance Tracking: Uses tqdm to provide visual progress bars for all time-consuming steps (Node Extraction, Simplification, and Iterative NPA Analysis).

Network Simplification: Automatically removes redundant degree-2 nodes to create a topologically clean network for analysis.

Scale Analysis: Supports calculation of measures across user-defined buffer distances (e.g., 0.25, 0.5, 1.0 miles).

🛠️ Setup and Execution

1. Requirements

This project requires the following Python libraries. You can install them using the provided requirements.txt:

pip install -r requirements.txt


2. Data Preparation

Place your required input files in the structure specified by the configuration in src/run_analysis.py:

| File Variable | Description | Default Path |
| source_edges_path | Global street centerline GeoDataFrame. | ../data/Streets_Meckl_10_8_2023/original_data/Streets.shp |
| NPA_shape_path | Study area (NPA) boundary GeoDataFrame. | ../data/QOL/prpcrmarea_Ctrl.shp |

Note: The output files (NPA_*.shp and Final_dataset.shp) will be created in the same directory as the input prpcrmarea_Ctrl.shp.

3. Running the Analysis

The entire pipeline is executed via a single script:

(Optional) Configure: Modify the buffer_list_mile variable in src/run_analysis.py to change the analysis scales.

Execute: Run the primary execution script from the root directory:

python src/run_analysis.py


The console output will display the progress bars for node extraction, network simplification, and analysis of each NPA.

📜 License

[Placeholder for your chosen license, e.g., MIT License]
