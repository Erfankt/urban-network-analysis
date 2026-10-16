"""
Interactive setup of the pipeline inputs, and helpers around the file that records
every interactive choice, data/final_dataset/user_selection.json:

    run_setup()           stage 1: input files, buffer distances, dependent variable, controls
    select_pca_method()   stage 2: extraction method
    load_selection()      read the saved choices (stage 2 reads them instead of re-prompting)
    update_selection()    add the stage-2 choices to the file

Every prompt shows its default in brackets; Enter accepts it. Answers can also be
given up front (command-line flags of run_analysis.py), in which case only the
missing ones are asked for.
"""
import os, json
import geopandas as gpd
from street_network_processor import StreetNetworkProcessor, NETWORK_METRICS

SELECTION_PATH = "../data/final_dataset/user_selection.json"
MAX_FIELD_LEN = 10          # shapefile field-name limit
STAGE1_KEYS = ("streets_path", "neighborhoods_path", "buffer_list_mile", "crime_col", "ctrl_vars")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _ask(prompt, default=None):
    shown = "" if default is None else f" [{default}]"
    ans = input(f"{prompt}{shown}: ").strip()
    return ans if ans else default


def _yes(prompt):
    return input(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")


def _numeric_columns(gdf):
    return [c for c in gdf.columns
            if c not in ("id", "geometry") and gdf[c].dtype.kind in "iuf"]


def _parse_choice(text, options):
    """Accept comma/space separated column numbers or names; return column names."""
    chosen = []
    for tok in str(text).replace(",", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(options):
            chosen.append(options[int(tok) - 1])
        elif tok in options:
            chosen.append(tok)
        else:
            raise ValueError(f"'{tok}' is not a valid column number or name")
    return list(dict.fromkeys(chosen))  # drop duplicates, keep order


def _read_layer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    return gpd.read_file(path)


# --------------------------------------------------------------------------
# input files
# --------------------------------------------------------------------------
def check_layers(edges, neighborhoods):
    """Return the list of reasons the two layers cannot be processed (empty = fine)."""
    problems = []
    for name, gdf in (("streets", edges), ("neighborhoods", neighborhoods)):
        if gdf.crs is None:
            problems.append(f"{name}: no coordinate reference system defined")
        elif not gdf.crs.is_projected:
            problems.append(f"{name}: CRS is not projected ({gdf.crs.to_string()})")
        else:
            unit = (gdf.crs.axis_info[0].unit_name or "").lower()
            if "foot" not in unit and "feet" not in unit:
                problems.append(f"{name}: CRS unit is '{unit}'; the pipeline assumes US feet "
                                "(buffers in miles, areas in acres)")
    if edges.crs is not None and neighborhoods.crs is not None and edges.crs != neighborhoods.crs:
        problems.append("the two layers have different CRS; reproject one of them")

    if "id" not in neighborhoods.columns:
        problems.append("neighborhoods: no 'id' column (unique neighborhood identifier)")
    else:
        ids = neighborhoods["id"]
        if ids.isna().any():
            problems.append("neighborhoods: 'id' has missing values")
        if ids.duplicated().any():
            problems.append(f"neighborhoods: 'id' is not unique ({int(ids.duplicated().sum())} duplicates)")

    kinds = set(edges.geom_type.unique())
    if not kinds <= {"LineString", "MultiLineString"}:
        problems.append(f"streets: expected line geometries, found {sorted(kinds)}")
    kinds = set(neighborhoods.geom_type.unique())
    if not kinds <= {"Polygon", "MultiPolygon"}:
        problems.append(f"neighborhoods: expected polygon geometries, found {sorted(kinds)}")
    return problems


def select_input_files(defaults, preset=None):
    """Ask for the two input shapefiles (or take them from preset), validate, return (streets, nbhd, gdf)."""
    preset = dict(preset or {})
    while True:
        streets = preset.get("streets_path") or _ask("Street centerline shapefile", defaults["streets_path"])
        nbhd = preset.get("neighborhoods_path") or _ask("Neighborhoods shapefile", defaults["neighborhoods_path"])
        try:
            edges, neighborhoods = _read_layer(streets), _read_layer(nbhd)
        except Exception as e:
            print(f"  -> {e}")
            preset = {}
            continue
        problems = check_layers(edges, neighborhoods)
        if problems:
            print("  -> the layers cannot be processed as they are:")
            for p in problems:
                print(f"     - {p}")
            preset = {}
            continue
        print(f"  streets: {len(edges)} segments | neighborhoods: {len(neighborhoods)} polygons "
              f"| CRS: {neighborhoods.crs.to_string()}")
        return streets, nbhd, neighborhoods


# --------------------------------------------------------------------------
# buffer distances
# --------------------------------------------------------------------------
def parse_buffers(text):
    """Comma/space separated distances in miles -> sorted list of floats, validated for column naming."""
    values = []
    for tok in str(text).replace(",", " ").split():
        try:
            v = float(tok)
        except ValueError:
            raise ValueError(f"'{tok}' is not a number")
        if v <= 0:
            raise ValueError("buffer distances must be positive")
        values.append(v)
    if not values:
        raise ValueError("enter at least one distance")
    values = sorted(set(values))
    for v in values:
        if v >= 1 and round(v * 10) != v * 10:
            raise ValueError(f"{v}: distances of 1 mile or more must be multiples of 0.1 "
                             "(the column suffix keeps one decimal)")
    codes = [StreetNetworkProcessor.buffer_code(v) for v in values]
    if len(set(codes)) != len(codes):
        raise ValueError("these distances produce duplicate column suffixes: "
                         + ", ".join(f"{v} -> {c}" for v, c in zip(values, codes)))
    longest = max(len(m) for m in NETWORK_METRICS)
    too_long = [v for v, c in zip(values, codes) if longest + len(c) > MAX_FIELD_LEN]
    if too_long:
        raise ValueError(f"{too_long}: the column names would exceed the {MAX_FIELD_LEN}-character "
                         "shapefile limit; use at most two decimals (e.g. 0.12 instead of 0.125)")
    return values


def select_buffers(default, preset=None):
    default_text = ", ".join(str(b) for b in default)
    while True:
        text = preset or _ask("Buffer distances around each neighborhood, in miles (comma-separated)", default_text)
        try:
            values = parse_buffers(text)
        except ValueError as e:
            print(f"  -> {e}")
            preset = None
            continue
        print("  column suffixes: " + ", ".join(f"{v} mi -> {StreetNetworkProcessor.buffer_code(v)}" for v in values))
        return values


# --------------------------------------------------------------------------
# dependent variable and controls
# --------------------------------------------------------------------------
def select_variables(neighborhoods, preset_crime=None, preset_controls=None):
    """Return (crime_col, ctrl_vars) chosen among the numeric columns of the neighborhoods layer."""
    gdf = _read_layer(neighborhoods) if isinstance(neighborhoods, str) else neighborhoods
    options = _numeric_columns(gdf)
    if not options:
        raise ValueError("the neighborhoods layer has no numeric columns besides 'id'")
    print("\nNumeric columns in the neighborhoods layer:")
    for i, c in enumerate(options, 1):
        print(f"  {i:2d}. {c}")

    while True:
        try:
            text = preset_crime or input("\nDependent variable (crime count), one number or name: ")
            crime = _parse_choice(text, options)
            if len(crime) != 1:
                raise ValueError("choose exactly one column")
            crime_col = crime[0]
            break
        except ValueError as e:
            print(f"  -> {e}")
            preset_crime = None

    while True:
        try:
            text = preset_controls or input("Control variables, numbers or names separated by commas/spaces: ")
            ctrl_vars = _parse_choice(text, options)
            if not ctrl_vars:
                raise ValueError("choose at least one column")
            if crime_col in ctrl_vars:
                raise ValueError("the dependent variable cannot also be a control")
            break
        except ValueError as e:
            print(f"  -> {e}")
            preset_controls = None
    return crime_col, ctrl_vars


# --------------------------------------------------------------------------
# stage-1 setup: everything the pipeline needs, saved to user_selection.json
# --------------------------------------------------------------------------
def _summary(sel):
    return (f"  streets             : {sel['streets_path']}\n"
            f"  neighborhoods       : {sel['neighborhoods_path']}\n"
            f"  buffers (miles)     : {', '.join(str(b) for b in sel['buffer_list_mile'])}\n"
            f"  dependent variable  : {sel['crime_col']}\n"
            f"  control variables   : {', '.join(sel['ctrl_vars'])}")


def run_setup(defaults, preset=None, reuse=False, selection_path=SELECTION_PATH):
    """
    Collect the stage-1 inputs: input files, buffer distances, dependent variable and
    controls. Values in `preset` (e.g. from command-line flags) are used without asking;
    everything else is prompted for, with `defaults` offered. If a complete selection was
    saved earlier and nothing is preset, it is offered for reuse (taken without asking
    when reuse=True). Returns the selection dict, which is also written to selection_path.
    """
    preset = {k: v for k, v in (preset or {}).items() if v}

    if os.path.exists(selection_path) and not preset:
        with open(selection_path) as f:
            prev = json.load(f)
        if all(k in prev for k in STAGE1_KEYS):
            print(f"\nPrevious selection found in {selection_path}:\n{_summary(prev)}")
            if reuse or _yes("Reuse it?"):
                missing = [p for p in (prev["streets_path"], prev["neighborhoods_path"]) if not os.path.exists(p)]
                if not missing:
                    return prev
                print(f"  -> input file(s) not found: {missing}; starting over")

    print("\n=== Pipeline inputs (Enter accepts the default in brackets) ===")
    streets, nbhd, gdf = select_input_files(defaults, preset)
    buffers = select_buffers(defaults["buffer_list_mile"], preset.get("buffers"))
    crime_col, ctrl_vars = select_variables(gdf, preset.get("crime_col"), preset.get("ctrl_vars"))

    sel = {"streets_path": streets, "neighborhoods_path": nbhd, "buffer_list_mile": buffers,
           "crime_col": crime_col, "ctrl_vars": ctrl_vars}
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump(sel, f, indent=2)       # a new stage-1 setup starts a fresh file: stage-2 choices must be redone
    print(f"\nSelection saved to {selection_path}:\n{_summary(sel)}")
    return sel


def load_selection(selection_path=SELECTION_PATH):
    """Read the saved selection (used by stage 2); raises if stage 1 has not been run."""
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            f"{selection_path} not found. Run 'python run_analysis.py' first to set up the inputs.")
    with open(selection_path) as f:
        sel = json.load(f)
    missing = [k for k in STAGE1_KEYS if k not in sel]
    if missing:
        raise KeyError(f"{selection_path} lacks {missing}; run 'python run_analysis.py' again.")
    return sel


def update_selection(selection_path=SELECTION_PATH, **fields):
    """Add or overwrite entries in the saved selection (e.g. the PCA choices made in stage 2)."""
    sel = {}
    if os.path.exists(selection_path):
        with open(selection_path) as f:
            sel = json.load(f)
    sel.update(fields)
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump(sel, f, indent=2)
    return sel


# --------------------------------------------------------------------------
# stage-2 prompt: extraction method
# --------------------------------------------------------------------------
PCA_METHODS = {
    "principal": "principal components (PCA in the strict sense, as the paper describes it)",
    "minres":    "minimum-residual common factor analysis (factor_analyzer default; what the original run used)",
    "ml":        "maximum-likelihood common factor analysis",
}


def select_pca_method():
    """Prompt for the extraction method used by pca_analysis; returns its name."""
    keys = list(PCA_METHODS)
    print("\nExtraction method for the component analysis:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k:10s} {PCA_METHODS[k]}")
    while True:
        ans = input("Choose a method (number or name): ").strip().lower()
        if ans.isdigit() and 1 <= int(ans) <= len(keys):
            return keys[int(ans) - 1]
        if ans in keys:
            return ans
        print("  -> choose one of: " + ", ".join(keys))
