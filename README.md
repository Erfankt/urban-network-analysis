# Street Network Configuration and Property Crime: data pipeline

Reproducible data-construction pipeline for the paper

> Kefayat, E., & Thill, J.-C. (2025). *Urban Street Network Configuration and Property Crime: An Empirical Multivariate Case Study.* ISPRS International Journal of Geo-Information, 14(5), 200. https://doi.org/10.3390/ijgi14050200

Starting from a cleaned street centerline layer and a set of neighborhood polygons, the pipeline builds the neighborhood-level street networks, measures their configuration at five nested spatial scales, reduces the measures (and the socio-spatial controls) with a component analysis, and writes a modeling-ready dataset with a data dictionary. The repository ends there: the spatial count models of the paper (negative binomial GLM, SAR-NB and CAR-NB estimated with INLA, GWR-NB) consume this dataset but are fitted outside the repository.

Every input and every analytical choice (input files, buffer distances, dependent variable, control variables, extraction method, number of components, representative measures) is entered interactively at run time, validated, and recorded in `data/final_dataset/user_selection.json`, so the pipeline can be rerun on another city or with other choices without editing code.

## Workflow

```
data/streetnetwork/streets.shp            data/neighborhoods/neighborhoods.shp
(cleaned street centerline)               (polygons: id, crime count, control variables)
            |                                        |
            v                                        v
  1. python run_analysis.py      - input files, buffer distances, dependent variable and
                                   controls (prompts with defaults, or command-line flags)
                                 - build the network, measure 8 metrics x 5 buffers per neighborhood
                                   (checkpointed in data/checkpoints/: an interrupted run resumes)
                                 -> data/final_dataset/extracted_measures.shp
            |
            v
  2. pca_analysis.ipynb  or  python pca_analysis.py
                                 - choose the extraction method (prompt)
                                 - street-network block: overview, inspect candidate solutions,
                                   choose the number of components, choose one representative
                                   measure per component (prompts)
                                 - socio-spatial block: overview, inspect, choose (prompts)
            |
            v
  3. final dataset               -> data/final_dataset/final_dataset.shp / .csv
                                 -> data/final_dataset/final_dataset_dictionary.csv
                                 -> data/final_dataset/user_selection.json
```

## Setup

Python 3.11 is recommended (the paper's computations used it). Install the dependencies with

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `scikit-learn<1.6` because `factor_analyzer` 0.5.x fails with newer scikit-learn releases. Jupyter is only needed for the notebook version of stage 2.

## Input data

The default locations are the ones below (other paths can be given at the prompt or with `--streets` / `--neighborhoods`; any `.shp` under `data/` is ignored by git). The **streets layer must be in a projected coordinate reference system in US feet** (e.g. a State Plane zone): the processor converts the buffer distances from miles to feet, street lengths to miles and areas to acres on that assumption. The streets define the working CRS; if the neighborhoods layer is in a different CRS it is reprojected to the streets CRS automatically, and the run prints a note saying so (your files are not modified). Two definitions of the same system, e.g. an EPSG code and the WKT written by desktop GIS software, are recognised as such (no coordinate moves when reprojecting) and the geometry is left untouched. The setup checks both layers before anything runs: both have a CRS, the streets CRS is projected and in feet, line and polygon geometries, and a unique `id` column.

| File | Content | Requirements |
|---|---|---|
| `data/streetnetwork/streets.shp` | Street centerlines (LineString or MultiLineString) | Already cleaned: the pipeline assumes non-functional segments (e.g. retired or non-service roads) have been removed. |
| `data/neighborhoods/neighborhoods.shp` | Neighborhood polygons | A unique identifier column named `id`; a numeric crime-count column; numeric columns for the control variables. Study-area decisions (e.g. the paper excluded the airport neighborhood) are made before running the pipeline. |

The paper's units of analysis are the 371 Neighborhood Profile Areas (NPAs) of the City of Charlotte, NC (see *Reproducing the Charlotte study* below).

## Stage 1: street-network measures (`src/run_analysis.py`)

```bash
cd src
python run_analysis.py
```

The script asks for everything it needs, showing a default in brackets that Enter accepts: the two input shapefiles (validated as described above), the buffer distances in miles (comma-separated; the paper's five are the default; the distances must be positive and distinct and must yield column names within the 10-character shapefile limit, so at most two decimals), then, from the numeric columns of the neighborhoods layer, the dependent variable (crime count) and the control variables, by number or name. The defaults live in `DEFAULTS` at the top of `run_analysis.py`.

The same answers can be given up front, for scripted or repeated runs; whatever is missing is still prompted for:

```bash
python run_analysis.py --streets ../data/streetnetwork/streets.shp --neighborhoods ../data/neighborhoods/neighborhoods.shp \
                       --buffers 0.1,0.25,0.5,0.75,1.0 --crime propcrimct --controls bachdg,med_age,emp
python run_analysis.py --reuse        # rerun with the previously saved answers, no questions
python run_analysis.py --recompute    # ignore the checkpoints (see below) and measure every buffer again
```

The answers are saved to `data/final_dataset/user_selection.json`; on the next run the script offers to reuse them. A new setup starts a fresh file, so the stage-2 choices are made again on the new inputs.

It then runs `StreetNetworkProcessor` (`src/street_network_processor.py`):

1. **Node extraction**: a node at every segment endpoint; shared endpoints become one node.
2. **Simplification**: degree-2 nodes (continuations of the same road) are removed and their segments merged, so nodes represent intersections and dead ends.
3. **Buffered local networks**: each neighborhood polygon is buffered by the chosen distances (the paper's 0.1, 0.25, 0.5, 0.75 and 1.0 miles by default) and the segments intersecting the buffer form the local network, of which the largest connected component is kept.
4. **Measures**: eight metrics per neighborhood and buffer, computed with NetworkX (betweenness and shortest paths are length-weighted).

| Metric code | Measure | Definition |
|---|---|---|
| `int_den` | Intersection density | Nodes of degree >= 3 per acre of the buffered polygon |
| `st_den` | Street density | Total street length per acre |
| `nd_deg` | Average node degree | Mean number of segments meeting at a node |
| `cl_cof` | Average clustering coefficient | Mean local clustering of the nodes |
| `shtpth` | Average shortest path | Mean length-weighted shortest path between node pairs, in miles |
| `bt_cnt` | Average betweenness centrality | Mean length-weighted betweenness of the nodes |
| `cl_cnt` | Average closeness centrality | Mean closeness of the nodes |
| `pr_cnt` | Average PageRank | Mean PageRank of the nodes |

Column names are `<metric code><buffer code>`, where the buffer code is the distance in miles with the decimal point removed: `1` (0.1 mi), `025`, `05`, `075`, `10` (1.0 mi); e.g. `int_den075`. The rule is implemented once, in `StreetNetworkProcessor.buffer_code()`, and keeps field names within the 10-character limit of shapefiles. Output: `data/final_dataset/extracted_measures.shp`, the neighborhoods layer with the measure columns appended (40 for the five default buffers).

The run can take a while for a city-sized network: the centrality and shortest-path measures are computed for every node of every local network, and progress bars report each step. The larger buffers take disproportionately longer, since these measures grow faster than the size of the local network.

**Checkpoints and resuming.** The measures are saved as they are produced, not at the end: each buffer has a CSV in `data/checkpoints/<fingerprint>/` to which a neighborhood's row is appended the moment it has been measured. A run that stops for any reason (an error, Ctrl+C, the computer going to sleep) therefore loses at most the neighborhood in progress; `python run_analysis.py --reuse` continues from there, skipping the neighborhoods and buffers already done. The checkpoints are keyed by a fingerprint of the street and neighborhood geometries (`inputs.json` in the folder says which files they came from), so a run with other layers never reuses them by mistake, while a run with the same layers reuses them even if the attributes changed: changing the crime column or the control variables, or adding a buffer distance, costs only the new work. `extracted_measures.shp` is assembled from the checkpoints at the end of every run and overwritten, since rebuilding it takes seconds. `--recompute` ignores the checkpoints of the buffers in the run (after a change to the measure code, say); deleting the folder has the same effect. The folder is not tracked by git.

## Stage 2: component analysis (`src/pca_analysis.ipynb` or `src/pca_analysis.py`)

The notebook and the script perform the same steps with the same functions (`src/common.py`, `src/user_inputs.py`, `src/final_dataset.py`); use whichever you prefer. The script saves every figure to `output/pca_diagnostics/` and, by default, also opens the component-analysis figures in windows so they can be examined while a prompt waits (close the windows to continue); `--no-windows` runs it headless and `--no-distribution-plots` skips the per-variable figures.

```bash
cd src
python pca_analysis.py                 # or open pca_analysis.ipynb in Jupyter
```

Steps:

1. Load `extracted_measures.shp` and the saved variable selection; an interactive map of the neighborhoods (HTML), descriptive statistics, a missing-value check and per-variable distribution plots. The component analysis cannot run on missing values (the correlation matrix is undefined), so the run stops at the check if any selected variable has them, listing the variables and the neighborhoods concerned. Fix them in the neighborhoods layer (fill or drop; the paper's choices are described under *Reproducing the Charlotte study*), rerun stage 1 with `--reuse`, which takes seconds because the network measures come from the checkpoints, and run stage 2 again.
2. **Extraction method** (prompt): `principal` (principal components, as the paper describes the procedure), `minres` (minimum-residual common factor analysis: the `factor_analyzer` default, which is what the original computation ran) or `ml`. Rotation is varimax, as in the paper.
3. **Street-network block**, 40 measures: (a) *overview*, independent of the number of components: correlation heatmap, scree plot of all eigenvalues with the Kaiser line and cumulative variance, eigenvalue table and rule-of-thumb counts; (b) *inspect*: type a number of components to see the full report for that solution (unrotated and rotated loadings as tables and heatmaps, variance explained, communalities, salient loadings per component, cross-loading variables, variables captured by no component, low communalities), as often as needed; (c) *final choice* of the number to retain; (d) **representative measures**: one original measure per retained component, chosen from the top-loading candidates (with their cross-loadings shown). Following the paper, these measures, not the component scores, are the street-network regressors, entered after a log(x + 1) transformation, which keeps the coefficients in interpretable units and preserves the buffer scale of each measure.
4. **Socio-spatial block**: the same overview, inspect and final choice; the retained component scores are the controls.

All choices are appended to `user_selection.json`, which after both stages reads like:

```json
{
  "streets_path": "../data/streetnetwork/streets.shp",
  "neighborhoods_path": "../data/neighborhoods/neighborhoods.shp",
  "buffer_list_mile": [0.1, 0.25, 0.5, 0.75, 1.0],
  "crime_col": "propcrimct",
  "ctrl_vars": ["bachdg", "med_age", "..."],
  "pca_method": "minres",
  "pca_rotation": "varimax",
  "pca_factors_network": 4,
  "pca_factors_ctrl": 9,
  "network_representatives": ["int_den075", "cl_cof075", "shtpth075", "bt_cnt05"]
}
```

## Stage 3: the final dataset

`data/final_dataset/final_dataset.shp` (with geometry, for spatial weights) and `final_dataset.csv` (the same table) contain one row per neighborhood:

| Column(s) | Role | Use in the models |
|---|---|---|
| `id` | identifier | |
| `cent_x`, `cent_y` | polygon centroid in the layer CRS | coordinates for GWR |
| crime column (name as selected) | dependent variable | property crime count |
| `area_acre` | exposure | `log(area_acre)` as offset in the count models |
| representative measures (original names, e.g. `int_den075`) | street-network regressors | apply `log(x + 1)` before modeling |
| `PC1_net` ... `PCn_net` | network component scores | alternative specification for a robustness check; not log-transformed |
| `PC1_ctrl` ... `PCk_ctrl` | socio-spatial component scores | controls |

`final_dataset_dictionary.csv` repeats this for the actual columns, including which component each representative stands for and its rotated loading. The paper's models use queen contiguity for the spatial weights, which can be built from the shapefile.

## Reproducing the Charlotte study

Data sources used in the paper: the Mecklenburg County street centerline layer (2023) for the network, and the Charlotte/Mecklenburg Quality of Life Explorer (2022) for the NPA polygons, the property-crime counts (Charlotte-Mecklenburg Police Department) and the socio-spatial variables (American Community Survey and Mecklenburg County tax parcel data). The NPA containing the airport was excluded, leaving 371 neighborhoods.

The 16 control variables of the paper (Table 4), with the field names used in the authors' neighborhood layer:

| Field | Variable |
|---|---|
| `bachdg` | Population 25+ with a bachelor's degree or higher (share) |
| `med_age` | Median age of residents |
| `emp` | Employment rate (civilian labor force aged 16-64) |
| `race_latin` | Hispanic or Latino residents (%) |
| `race_asian` | Asian residents (%) |
| `race_black` | Black residents (%) |
| `race_white` | White residents (%) |
| `income` | Inflation-adjusted median household income |
| `home_own` | Owner-occupied share of occupied housing units |
| `resoccup` | Occupied share of all housing units |
| `sinfamrate` | Single-family units / total housing units |
| `sinfamage` | Mean age of single-family structures |
| `hs_den` | Housing units per acre |
| `sinfamsize` | Mean heated square footage of single-family units |
| `salesprice` | Mean sales price of single-family dwellings, condominiums and townhomes (two fiscal years) |
| `heatedarea` | Total heated square footage of all structures |

The Quality of Life Explorer export has no value for some variables in 23 of the 371 NPAs (40 cells in all), where there are no single-family homes or sales to report, or the Census estimate is suppressed. The paper's modelling table carries the following values in those cells, and the neighborhoods layer used with this repository has them filled in the same way:

| Variable | NPAs | Value |
|---|---|---|
| `sinfamage` | 15 | 0 (no single-family units) |
| `sinfamsize` | 13 | 0 (no single-family units) |
| `salesprice` | 3 | 0 (no sales) |
| `sinfamrate`, `home_own` | 1 each | 0 |
| `income` | 6 (ids 28, 193, 247, 293, 294, 315) | 250001, 250001, 50000, 36786, 51714, 56549 (ACS estimates; 250001 is the top-coded value) |
| `race_white` | 1 (id 380) | 77.8 |

Choices made in the paper: five buffers (0.1 to 1.0 mi); a four-component solution for the 40 network measures (74.1% of their variance), represented by intersection density, average clustering coefficient and average shortest path at 0.75 mi and average betweenness centrality at 0.5 mi; a nine-component solution for the 16 controls. The paper describes the reduction as PCA with varimax rotation; the original computation used `factor_analyzer` with its default `minres` extraction and a varimax rotation, which is why both options are offered.

## Repository structure

```
.
|-- data/
|   |-- streetnetwork/      streets.shp (input; not tracked)
|   |-- neighborhoods/      neighborhoods.shp (input; not tracked)
|   |-- checkpoints/        one folder per pair of input layers, one CSV of measures per buffer (generated; not tracked)
|   `-- final_dataset/      extracted_measures.shp, final_dataset.*, user_selection.json (generated; not tracked)
|-- output/pca_diagnostics/ figures and the neighborhood map written by pca_analysis.py (generated; not tracked)
|-- src/
|   |-- run_analysis.py              stage 1 entry point: prompts or flags for the inputs, DEFAULTS
|   |-- street_network_processor.py  StreetNetworkProcessor: network construction and measures
|   |-- user_inputs.py               prompts, input validation and user_selection.json helpers
|   |-- common.py                    component analysis: factor_loading_matrices, explore_pca, select_representatives
|   |-- final_dataset.py             assembly of the final dataset and its data dictionary
|   |-- pca_analysis.ipynb           stage 2 and 3, notebook version
|   `-- pca_analysis.py              stage 2 and 3, script version
|-- requirements.txt
`-- README.md
```

## License

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

## Citation

If you use this code, please cite the paper:

```
Kefayat, E., & Thill, J.-C. (2025). Urban Street Network Configuration and Property Crime:
An Empirical Multivariate Case Study. ISPRS International Journal of Geo-Information, 14(5), 200.
https://doi.org/10.3390/ijgi14050200
```

## Contact

Questions and collaboration: open an issue or email ekefayat@charlotte.edu
