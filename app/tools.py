"""
What the assistant can look up: a few read-only tools over the data of the current run.

The model never runs code and never writes anything. It can ask for a fixed list of questions
(the declarations in TOOLS below); each is answered here from the files on disk, with the column
names checked against the real columns, and the answer goes back to the model as text. A tool may
also return a table, which the chat panel shows (and offers as a download) but does not send to the
model, so that "the area of every neighborhood" is complete in the panel while the model reads a
summary and the extremes.

`ctx` is a dict made by the app: neighborhoods_path, streets_path, id_col, measures_path, final_csv
(any may be missing). Relative paths are taken from the src/ folder, as everywhere in the app.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
MAX_TEXT = 3500            # characters of tool output sent to the model
SOURCES = ("neighborhoods", "measures", "final")

TOOLS = [
    {"name": "list_data",
     "description": "List the data of the current run: for each of the neighborhoods layer, the stage-1 "
                    "measures table and the final dataset, whether it exists, its number of rows and its "
                    "column names. Call this first when you do not know which columns exist.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "neighborhood_areas",
     "description": "Compute the area of every neighborhood from the polygons (not from any column). Returns "
                    "summary statistics and the largest and smallest neighborhoods; the full table is shown "
                    "to the user in the chat. This is the same quantity as the column area_acre of the "
                    "final dataset when the unit is acres.",
     "parameters": {"type": "object", "properties": {
         "unit": {"type": "string", "enum": ["acres", "square_miles", "square_km"],
                  "description": "Unit of the area (default acres)."},
         "limit": {"type": "integer", "description": "How many of the largest and smallest to list (default 5)."}}}},
    {"name": "column_summary",
     "description": "Summary of one column: count, missing values, mean, standard deviation, min, quartiles, "
                    "max, and the neighborhoods holding the extreme values. For a text column, the most "
                    "frequent values.",
     "parameters": {"type": "object", "properties": {
         "source": {"type": "string", "enum": list(SOURCES), "description": "Which table the column is in."},
         "column": {"type": "string", "description": "Exact column name (see list_data)."}},
         "required": ["source", "column"]}},
    {"name": "compare_columns",
     "description": "Pearson and Spearman correlation between two numeric columns of the same table, and the "
                    "number of neighborhoods used.",
     "parameters": {"type": "object", "properties": {
         "source": {"type": "string", "enum": list(SOURCES)},
         "column_a": {"type": "string"}, "column_b": {"type": "string"}},
         "required": ["source", "column_a", "column_b"]}},
    {"name": "get_neighborhoods",
     "description": "The values of chosen columns for chosen neighborhoods (at most 20), by identifier value.",
     "parameters": {"type": "object", "properties": {
         "source": {"type": "string", "enum": list(SOURCES)},
         "ids": {"type": "array", "items": {"type": "string"}, "description": "Identifier values, as text."},
         "columns": {"type": "array", "items": {"type": "string"}, "description": "At most 12 column names."}},
         "required": ["source", "ids", "columns"]}},
]


# ---------------------------------------------------------------------------
# Reading the files (cached by modification time)
# ---------------------------------------------------------------------------
_CACHE = {}


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (SRC_DIR / p).resolve()


def _read(path, geometry):
    p = _resolve(path)
    if not p.exists():
        return None
    key = (str(p), p.stat().st_mtime, geometry)
    if key not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        if p.suffix.lower() == ".csv":
            _CACHE[key] = pd.read_csv(p)
        else:
            import geopandas as gpd
            gdf = gpd.read_file(p)
            _CACHE[key] = gdf if geometry else pd.DataFrame(gdf.drop(columns="geometry"))
    return _CACHE[key]


def _table(ctx, source, geometry=False):
    """The requested table as a DataFrame, or (None, reason)."""
    key = {"neighborhoods": "neighborhoods_path", "measures": "measures_path", "final": "final_csv"}.get(source)
    if key is None:
        return None, f"Unknown source '{source}'. Use one of: {', '.join(SOURCES)}."
    path = ctx.get(key)
    if not path:
        hint = {"neighborhoods": "The user has not chosen the neighborhoods layer yet (Inputs section).",
                "measures": "The stage-1 measures do not exist yet (Stage 1 section).",
                "final": "The final dataset has not been built yet (Final dataset section)."}[source]
        return None, hint
    df = _read(path, geometry and source != "final")
    if df is None:
        return None, f"The {source} file is not on disk yet."
    return df, None


def _check_column(df, column):
    if column in df.columns:
        return None
    close = [c for c in df.columns if c.lower() == str(column).lower()]
    if close:
        return None
    return f"There is no column '{column}'. Columns: {', '.join(map(str, df.columns[:60]))}"


def _col(df, column):
    if column in df.columns:
        return column
    return next(c for c in df.columns if c.lower() == str(column).lower())


def _fmt(x):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "nan"
        return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4g}"
    return str(x)


def _clip(text):
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "\n… (shortened)"


# ---------------------------------------------------------------------------
# The tools: each returns {"text": ..., "table": DataFrame or None}
# ---------------------------------------------------------------------------

def list_data(ctx):
    lines = []
    for source in SOURCES:
        df, why = _table(ctx, source)
        if df is None:
            lines.append(f"- {source}: not available. {why}")
        else:
            lines.append(f"- {source}: {len(df)} rows; columns: {', '.join(map(str, df.columns))}")
    lines.append(f"Neighborhood identifier column: {ctx.get('id_col', 'id')}")
    return {"text": _clip("\n".join(lines)), "table": None}


def neighborhood_areas(ctx, unit="acres", limit=5):
    gdf, why = _table(ctx, "neighborhoods", geometry=True)
    if gdf is None:
        return {"text": why, "table": None}
    unit = unit if unit in ("acres", "square_miles", "square_km") else "acres"
    limit = max(1, min(int(limit or 5), 15))
    try:
        if str(ctx.get("streets_path") or ""):               # same CRS as the pipeline works in
            import geopandas as gpd
            sp = _resolve(ctx["streets_path"])
            if sp.exists():
                crs = gpd.read_file(sp, rows=1).crs
                if crs is not None and gdf.crs is not None and gdf.crs != crs:
                    gdf = gdf.to_crs(crs)
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from final_dataset import area_in_acres
        acres = area_in_acres(gdf)
    except Exception as e:
        return {"text": f"The area could not be computed: {e}", "table": None}
    factor = {"acres": 1.0, "square_miles": 1 / 640.0, "square_km": 0.00404685642}[unit]
    area = acres * factor
    id_col = ctx.get("id_col", "id")
    ids = gdf[id_col] if id_col in gdf.columns else pd.Series(range(len(gdf)))
    table = pd.DataFrame({id_col: ids.values, f"area_{unit}": area.values.round(3)})
    s = area.describe()
    order = table.sort_values(f"area_{unit}", ascending=False)
    text = (f"Area of {len(table)} neighborhoods in {unit} (computed from the polygons in the layer CRS"
            f"{': ' + str(gdf.crs.name) if gdf.crs is not None else ''}).\n"
            f"total {_fmt(area.sum())}; mean {_fmt(s['mean'])}; median {_fmt(s['50%'])}; "
            f"min {_fmt(s['min'])}; max {_fmt(s['max'])}; std {_fmt(s['std'])}.\n"
            f"Largest {limit}: " + "; ".join(f"{r[0]} = {_fmt(r[1])}" for r in order.head(limit).itertuples(index=False))
            + f"\nSmallest {limit}: " + "; ".join(f"{r[0]} = {_fmt(r[1])}"
                                                  for r in order.tail(limit).iloc[::-1].itertuples(index=False))
            + "\nThe complete table is shown to the user in the chat, with a download button.")
    return {"text": _clip(text), "table": table.sort_values(id_col).reset_index(drop=True)}


def column_summary(ctx, source, column):
    df, why = _table(ctx, source)
    if df is None:
        return {"text": why, "table": None}
    bad = _check_column(df, column)
    if bad:
        return {"text": bad, "table": None}
    column = _col(df, column)
    s = df[column]
    id_col = ctx.get("id_col", "id")
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna()
        q = v.quantile([.25, .5, .75])
        text = (f"{source}.{column}: n={len(v)} of {len(s)} (missing {int(s.isna().sum())}, zeros {int((v == 0).sum())}); "
                f"mean {_fmt(v.mean())}, std {_fmt(v.std())}, min {_fmt(v.min())}, Q1 {_fmt(q[.25])}, "
                f"median {_fmt(q[.5])}, Q3 {_fmt(q[.75])}, max {_fmt(v.max())}.")
        if id_col in df.columns and len(v):
            top, low = df.loc[v.nlargest(3).index, id_col], df.loc[v.nsmallest(3).index, id_col]
            text += (f" Highest: " + "; ".join(f"{i} = {_fmt(s[j])}" for j, i in top.items())
                     + ". Lowest: " + "; ".join(f"{i} = {_fmt(s[j])}" for j, i in low.items()) + ".")
    else:
        vc = s.value_counts().head(8)
        text = (f"{source}.{column}: text column, {s.nunique()} distinct of {len(s)} (missing {int(s.isna().sum())}). "
                "Most frequent: " + "; ".join(f"{k} ({n})" for k, n in vc.items()))
    return {"text": _clip(text), "table": None}


def compare_columns(ctx, source, column_a, column_b):
    df, why = _table(ctx, source)
    if df is None:
        return {"text": why, "table": None}
    for c in (column_a, column_b):
        bad = _check_column(df, c)
        if bad:
            return {"text": bad, "table": None}
    a, b = _col(df, column_a), _col(df, column_b)
    if not (pd.api.types.is_numeric_dtype(df[a]) and pd.api.types.is_numeric_dtype(df[b])):
        return {"text": "Both columns must be numeric.", "table": None}
    if a == b:
        return {"text": "Those are the same column; give two different columns.", "table": None}
    pair = df[[a, b]].dropna()
    if len(pair) < 3:
        return {"text": "Fewer than 3 complete pairs.", "table": None}
    text = (f"{source}: {a} vs {b} over {len(pair)} neighborhoods: Pearson r = {pair[a].corr(pair[b]):.3f}, "
            f"Spearman rho = {pair[a].corr(pair[b], method='spearman'):.3f}.")
    return {"text": text, "table": None}


def get_neighborhoods(ctx, source, ids, columns):
    df, why = _table(ctx, source)
    if df is None:
        return {"text": why, "table": None}
    id_col = ctx.get("id_col", "id")
    if id_col not in df.columns:
        return {"text": f"The table has no identifier column '{id_col}'.", "table": None}
    columns = list(columns or [])[:12]
    for c in columns:
        bad = _check_column(df, c)
        if bad:
            return {"text": bad, "table": None}
    cols = [_col(df, c) for c in columns]
    want = [str(i) for i in (ids or [])][:20]
    sub = df[df[id_col].astype(str).isin(want)][[id_col] + cols]
    missing = [i for i in want if i not in set(sub[id_col].astype(str))]
    text = sub.round(3).to_string(index=False) if len(sub) else "No neighborhood matches."
    if missing:
        text += f"\nNot found: {', '.join(missing)}"
    return {"text": _clip(text), "table": sub.reset_index(drop=True) if len(sub) > 1 else None}


_FUNCS = {"list_data": list_data, "neighborhood_areas": neighborhood_areas, "column_summary": column_summary,
          "compare_columns": compare_columns, "get_neighborhoods": get_neighborhoods}


def call_tool(name, args, ctx):
    """Run one tool by name with the model's arguments; never raises. Returns {"text", "table"}."""
    fn = _FUNCS.get(name)
    if fn is None:
        return {"text": f"Unknown tool '{name}'. Available: {', '.join(_FUNCS)}.", "table": None}
    try:
        return fn(ctx, **(args or {}))
    except TypeError as e:
        return {"text": f"Wrong arguments for {name}: {e}", "table": None}
    except Exception as e:                                        # a tool failure goes back to the model as text
        return {"text": f"{name} failed: {e}", "table": None}
