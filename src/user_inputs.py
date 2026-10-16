"""
Interactive selection of the dependent variable and the socio-spatial control
variables from the neighborhoods layer. The choice is saved to
data/final_dataset/user_selection.json so pca_analysis.ipynb / pca_analysis.py can
reuse it without re-prompting.
"""
import os, json
import geopandas as gpd

SELECTION_PATH = "../data/final_dataset/user_selection.json"


def _numeric_columns(gdf):
    return [c for c in gdf.columns
            if c not in ("id", "geometry") and gdf[c].dtype.kind in "iuf"]


def _parse_choice(text, options):
    """Accept comma/space separated column numbers or names; return column names."""
    chosen = []
    for tok in text.replace(",", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(options):
            chosen.append(options[int(tok) - 1])
        elif tok in options:
            chosen.append(tok)
        else:
            raise ValueError(f"'{tok}' is not a valid column number or name")
    return list(dict.fromkeys(chosen))  # drop duplicates, keep order


def select_variables(NPA_shape_path, selection_path=SELECTION_PATH):
    """Prompt for crime column and control variables; return (crime_col, ctrl_vars)."""
    if os.path.exists(selection_path):
        with open(selection_path) as f:
            prev = json.load(f)
        print(f"\nPrevious selection found in {selection_path}:")
        print(f"  dependent variable : {prev['crime_col']}")
        print(f"  control variables  : {', '.join(prev['ctrl_vars'])}")
        if input("Reuse it? [Y/n] ").strip().lower() in ("", "y", "yes"):
            return prev["crime_col"], prev["ctrl_vars"]

    gdf = gpd.read_file(NPA_shape_path)
    options = _numeric_columns(gdf)
    print("\nNumeric columns in the neighborhoods layer:")
    for i, c in enumerate(options, 1):
        print(f"  {i:2d}. {c}")

    while True:
        try:
            crime_col = _parse_choice(input("\nDependent variable (crime count), one number or name: "), options)
            if len(crime_col) != 1:
                raise ValueError("choose exactly one column")
            crime_col = crime_col[0]
            break
        except ValueError as e:
            print(f"  -> {e}")

    while True:
        try:
            ctrl_vars = _parse_choice(
                input("Control variables, numbers or names separated by commas/spaces: "), options)
            if not ctrl_vars:
                raise ValueError("choose at least one column")
            if crime_col in ctrl_vars:
                raise ValueError("the dependent variable cannot also be a control")
            break
        except ValueError as e:
            print(f"  -> {e}")

    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump({"crime_col": crime_col, "ctrl_vars": ctrl_vars}, f, indent=2)
    print(f"Selection saved to {selection_path}")
    return crime_col, ctrl_vars


def load_selection(selection_path=SELECTION_PATH):
    """Read a saved selection (used by the notebook); raises if none exists."""
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            f"{selection_path} not found. Run 'python run_analysis.py' first to choose variables.")
    with open(selection_path) as f:
        sel = json.load(f)
    return sel["crime_col"], sel["ctrl_vars"]


PCA_METHODS = {
    "principal": "principal components (PCA in the strict sense, as the paper describes it)",
    "minres":    "minimum-residual common factor analysis (factor_analyzer default; what the original run used)",
    "ml":        "maximum-likelihood common factor analysis",
}


def select_pca_method():
    """Prompt for the extraction method used by pca_analysis.ipynb; returns its name."""
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


def update_selection(selection_path=SELECTION_PATH, **fields):
    """Add or overwrite entries in the saved selection (e.g. the PCA choices made in the notebook)."""
    sel = {}
    if os.path.exists(selection_path):
        with open(selection_path) as f:
            sel = json.load(f)
    sel.update(fields)
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    with open(selection_path, "w") as f:
        json.dump(sel, f, indent=2)
    return sel
