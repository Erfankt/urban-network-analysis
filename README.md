# Street Network Configuration and Property Crime: data pipeline

Reproducible data-construction pipeline for the paper

> Kefayat, E., & Thill, J.-C. (2025). *Urban Street Network Configuration and Property Crime: An Empirical Multivariate Case Study.* ISPRS International Journal of Geo-Information, 14(5), 200. https://doi.org/10.3390/ijgi14050200

Starting from a cleaned street centerline layer and a set of neighborhood polygons, the pipeline builds the neighborhood-level street networks, measures their configuration at five nested spatial scales, reduces the measures (and the socio-spatial controls) with a component analysis, and writes a modeling-ready dataset with a data dictionary. The repository ends there: the spatial count models of the paper (negative binomial GLM, SAR-NB and CAR-NB estimated with INLA, GWR-NB) consume this dataset but are fitted outside the repository.

Every input and every analytical choice (input files, buffer distances, dependent variable, control variables, extraction method, number of components, representative measures) is entered interactively at run time, validated, and recorded in `data/final_dataset/user_selection.json`, so the pipeline can be rerun on another city or with other choices without editing code.

The pipeline has two front ends over the same code: the terminal scripts described below, and a **web interface** (`app/`, see *Web interface*) that runs the same stages from the browser, with an assistant beside every step that explains the diagnostics and interprets the run's own numbers.

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

Python 3.11 is recommended (the paper's computations used it). The dependencies live in a virtual environment, `.venv`, in the repository folder. It is created and filled once: if `.venv` already exists with the requirements installed (the check below tells), skip steps 1 and 3 and only activate it in each new terminal.

1. Create the environment (first time only):

```bash
python -m venv .venv
```

2. Activate it (in every new terminal; the prompt then starts with `(.venv)`):

```bash
# Windows, Command Prompt
.venv\Scripts\activate
# Windows, PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

3. Install the dependencies (first time, or after `requirements.txt` changes):

```bash
pip install -r requirements.txt
```

To see whether an existing environment is already complete, activate it and run `python -c "import geopandas, networkx, scipy, tqdm, factor_analyzer, matplotlib, seaborn, folium, sklearn; print(sklearn.__version__)"`: if it prints a version below 1.6 without an error, skip step 3. `requirements.txt` pins `scikit-learn<1.6` because `factor_analyzer` 0.5.x fails with newer scikit-learn releases, so an error such as `check_array() got an unexpected keyword argument 'force_all_finite'` means the script is running outside this environment, with a newer scikit-learn: activate the environment and run it again. Jupyter is only needed for the notebook version of stage 2; start it from the activated environment. The last group of `requirements.txt` (Streamlit and three small packages) is only needed for the web interface.

## Input data

The default locations are the ones below (other paths can be given at the prompt or with `--streets` / `--neighborhoods`; any `.shp` under `data/` is ignored by git). The **streets layer must be in a projected coordinate reference system in US feet** (e.g. a State Plane zone): the processor converts the buffer distances from miles to feet, street lengths to miles and areas to acres on that assumption. The streets define the working CRS; if the neighborhoods layer is in a different CRS it is reprojected to the streets CRS automatically, and the run prints a note saying so (your files are not modified). Two definitions of the same system, e.g. an EPSG code and the WKT written by desktop GIS software, are recognised as such (no coordinate moves when reprojecting) and the geometry is left untouched. The setup checks both layers before anything runs: both have a CRS, the streets CRS is projected and in feet, line and polygon geometries, and an identifier column with a distinct, non-missing value in every polygon. That column is chosen at setup (`id` by default, or `--id-col`); it keys the checkpoints and is kept under its own name in every output table.

| File | Content | Requirements |
|---|---|---|
| `data/streetnetwork/streets.shp` | Street centerlines (LineString or MultiLineString) | Already cleaned: the pipeline assumes non-functional segments (e.g. retired or non-service roads) have been removed. |
| `data/neighborhoods/neighborhoods.shp` | Neighborhood polygons | A unique identifier column (`id` by default; any column can be chosen); a numeric crime-count column; numeric columns for the control variables. Study-area decisions (e.g. the paper excluded the airport neighborhood) are made before running the pipeline. |

The paper's units of analysis are the 371 Neighborhood Profile Areas (NPAs) of the City of Charlotte, NC (see *Reproducing the Charlotte study* below).

## Stage 1: street-network measures (`src/run_analysis.py`)

```bash
cd src
python run_analysis.py
```

The script asks for everything it needs, showing a default in brackets that Enter accepts: the two input shapefiles (validated as described above), the neighborhood identifier column (`id` when the layer has one that is unique and complete, otherwise the first column that is), the buffer distances in miles (comma-separated; the paper's five are the default; the distances must be positive and distinct and must yield column names within the 10-character shapefile limit, so at most two decimals), then, from the numeric columns of the neighborhoods layer, the dependent variable (crime count) and the control variables, by number or name. The defaults live in `DEFAULTS` at the top of `run_analysis.py`.

The same answers can be given up front, for scripted or repeated runs; whatever is missing is still prompted for:

```bash
python run_analysis.py --streets ../data/streetnetwork/streets.shp --neighborhoods ../data/neighborhoods/neighborhoods.shp \
                       --id-col id --buffers 0.1,0.25,0.5,0.75,1.0 --crime propcrimct --controls bachdg,med_age,emp
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

**Checkpoints and resuming.** The measures are saved as they are produced, not at the end: each buffer has a CSV in `data/checkpoints/<fingerprint>/` to which a neighborhood's row is appended the moment it has been measured. A run that stops for any reason (an error, Ctrl+C, the computer going to sleep) therefore loses at most the neighborhood in progress; `python run_analysis.py --reuse` continues from there, skipping the neighborhoods and buffers already done. The checkpoints are keyed by a fingerprint of the street and neighborhood geometries (`inputs.json` in the folder says which files they came from), so a run with other layers never reuses them by mistake, while a run with the same layers reuses them even if the attributes changed: changing the crime column or the control variables, or adding a buffer distance, costs only the new work. `extracted_measures.shp` is assembled from the checkpoints at the end of every run and overwritten, since rebuilding it takes seconds. `--recompute` deletes the checkpoints of the buffers in the run before measuring (after a change to the measure code, say), so a recompute run that is interrupted and then resumed with `--reuse` never falls back on old files; deleting the folder has the same effect. The folder is not tracked by git.

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
  "id_col": "id",
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
| identifier column (name as selected, `id` in the paper's data) | identifier | |
| `cent_x`, `cent_y` | polygon centroid in the layer CRS | coordinates for GWR |
| crime column (name as selected) | dependent variable | property crime count |
| `area_acre` | exposure | `log(area_acre)` as offset in the count models |
| representative measures (original names, e.g. `int_den075`) | street-network regressors | apply `log(x + 1)` before modeling |
| `PC1_net` ... `PCn_net` | network component scores | alternative specification for a robustness check; not log-transformed |
| `PC1_ctrl` ... `PCk_ctrl` | socio-spatial component scores | controls |

`final_dataset_dictionary.csv` repeats this for the actual columns, including which component each representative stands for and its rotated loading. The paper's models use queen contiguity for the spatial weights, which can be built from the shapefile.

## Web interface (`app/streamlit_app.py`)

The same three stages from the browser, for users who would rather click than type. From the repository folder, in the activated environment (the web dependencies are in `requirements.txt`):

```bash
streamlit run app/streamlit_app.py
```

**Shortest start.** The app can set its own environment up. With Python installed, from the repository folder: `python -m venv .venv`, activate it (as in *Setup* above), `pip install streamlit`, then `streamlit run app/streamlit_app.py`. The first section, **0 · Setup**, checks the Python version, the `.venv` and every package of `requirements.txt` with its pinned version (`scikit-learn<1.6`). When all is in place it shows a one-line note and the page goes on; when something is missing it shows what, and a button installs it (into the running environment, or into a new or existing `.venv` after which a **Restart the app** button starts it again inside that environment and the page reloads by itself). Jupyter, needed only for the notebook, is an optional tick.

The app opens at `http://localhost:8501` and runs entirely on your computer; nothing is uploaded anywhere. The look follows the system's light or dark setting; the app's menu (top right, Settings, Theme) switches it, and the palette is defined in `.streamlit/config.toml`. It is one page with a setup note and four numbered sections that follow the pipeline (the sidebar links to each; a later section says what is missing until the earlier ones are done):

1. **Inputs**: each layer comes either from the data folder of the repository (`data/streetnetwork/`, `data/neighborhoods/`; every `.shp`, `.gpkg` and `.geojson` there is listed) or from your computer (a shapefile's parts, a `.zip`, a `.gpkg` or a `.geojson`, copied into that folder and then listed like any other). The layers load as soon as they are picked and are validated as in the terminal; the neighborhood identifier column is chosen here. When the two layers use different coordinate reference systems the page says so and offers to reproject the neighborhoods into the streets' system (a copy is saved beside the original; without it the run reprojects on the fly). A panel per dataset gives the counts, lengths or areas, CRS and columns with a histogram, and a map shows both layers, each switchable, with the attributes of a clicked feature, a choice of basemaps (OpenStreetMap by default) and a full-screen button.
2. **Stage 1**: buffer distances, dependent variable and controls, then *Run*. The run is `run_analysis.py` started in the background with all answers given, so it leaves exactly the files a terminal run leaves; the page shows one progress bar per buffer, read from the checkpoint CSVs, and the log. *Stop* ends the process and *Resume* continues it from the checkpoints; a browser refresh does not lose the run.
3. **Stage 2**: the extraction method, then for each block the overview (correlation heatmap, scree plot, eigenvalue table, rules of thumb), any candidate solution on request (rotated and unrotated loadings, variance explained, communalities, salient and cross-loading variables), the number to retain, and, for the street-network block, one dropdown per component for the representative measure, with its loading and cross-loading. A tab shows the distribution figure of any variable, and the last tab builds the final dataset.
4. **Final dataset**: the dictionary, a preview of the table, downloads (CSV, dictionary, zipped shapefile, the choices as JSON) and the saved figures.

The sections call `app/pipeline.py`, which calls the functions in `src/` from the same working directory as the terminal scripts, so an app run and a terminal run with the same choices write identical files to `data/final_dataset/` and `output/pca_diagnostics/`, including `user_selection.json` and the numbered figures.

**Cleaning up.** When the app shuts down normally (Ctrl+C in the terminal), it empties the `output/` folder: the stage-1 log and run record and the diagnostic figures, which are all regenerated when needed. The folder itself, `data/` (the measures, the checkpoints, the final dataset) and everything else stay; a stage-1 run that is still going keeps its log. Set `KEEP_OUTPUT=1` in the environment to keep `output/` as it is. Closing the terminal window instead of pressing Ctrl+C skips the cleanup.

**The assistant.** A chat panel beside the page explains the steps, the measures and the diagnostics, and interprets the current run: each question is sent with the method notes (what the pipeline computes and what the paper chose) and a description of the run so far, from the inputs and the stage-1 status to the eigenvalue table, the solutions inspected with their salient loadings, the candidates for the representatives and the choices made. It can suggest how many components to keep or which measure to pick, with its reasons; every choice is still made by clicking in the pages. It can also look things up in your data through a few read-only tools (`app/tools.py`): list the columns, compute the area of every neighborhood, summarize or correlate columns, fetch the values of given neighborhoods; a table it produces appears in the chat with a download button. It cannot run code, change files or choose for you (set `ASSISTANT_TOOLS=0` in `.env` to turn the lookups off). With no model connected, the panel offers three ways to connect one by clicking, with no file to edit: paste a free Gemini or Groq key (the app checks it and keeps it in `.env`), or, for a local model, install Ollama and press the button that downloads the model. The model is free by default: a local [Ollama](https://ollama.com) server with its `llama3.2` model (install Ollama, run `ollama pull llama3.2` once; a few GB of memory, slower without a GPU). A Gemini or Groq API key in a `.env` file switches to that provider instead; copy `.env.example` to `.env` to see the settings. With no model reachable the panel says how to set one up and the rest of the app works as before. Answers from a small local model are a guide to the diagnostics, not a substitute for reading them.

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
|-- output/app/             the stage-1 log and run record of the web app (generated; not tracked)
|-- app/
|   |-- streamlit_app.py             web interface entry point: the one page, its sidebar, the chat panel beside it
|   |-- pipeline.py                  the app's only link to src/: inputs, the stage-1 process, stage 2, the final dataset
|   |-- chat.py                      the assistant: provider settings (.env), the model calls, what it is told
|   |-- setup_env.py                 stage 0: checks the Python environment and installs what is missing
|   |-- tools.py                     the assistant's read-only data lookups (areas, summaries, correlations)
|   |-- chat_panel.py                the chat panel
|   |-- theme.py                     the stylesheet, title bar, section headers, sidebar stepper and footer
|   `-- views/                       one module per section: inputs, stage1, stage2, final
|-- src/
|   |-- run_analysis.py              stage 1 entry point: prompts or flags for the inputs, DEFAULTS
|   |-- street_network_processor.py  StreetNetworkProcessor: network construction and measures
|   |-- user_inputs.py               prompts, input validation and user_selection.json helpers
|   |-- common.py                    component analysis: computations and figures (overview_stats, fit_solution, ...) and the terminal workflow (explore_pca, select_representatives, ...)
|   |-- final_dataset.py             assembly of the final dataset and its data dictionary
|   |-- pca_analysis.ipynb           stage 2 and 3, notebook version
|   `-- pca_analysis.py              stage 2 and 3, script version
|-- .env.example            settings of the assistant (copy to .env)
|-- .streamlit/config.toml  display settings of the web app
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
