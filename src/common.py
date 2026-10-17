import functools, itertools, re
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt,seaborn as sns
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

try:
    from IPython.display import display
except ImportError:  # plain Python session, no notebook
    display = print


def factor_loading_matrices(
    data,
    variables,
    n_factors,
    rotation='varimax',
    plot=True,
    verbose=True,
    method='minres',
    show_scree=True
):
    """
    Computes factor analysis with unrotated and rotated solutions, returns loadings,
    communalities, eigenvalues, factor scores, and optionally plots diagnostic figures.

    This version is verbose and prints detailed step-by-step progress.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing variables used in factor analysis.
    variables : list
        Names of variables to include.
    n_factors : int
        Number of factors/components to extract.
    rotation : str or None
        Rotation method ('varimax', 'promax', None).
    plot : bool
        Whether to produce scree plot and heatmaps.
    verbose : bool
        Print detailed progress messages.
    method : str
        Extraction method passed to FactorAnalyzer: 'minres' (minimum residual,
        the library default and what the original analysis ran), 'ml' or
        'principal' (principal components).
    show_scree : bool
        When plot=True, also draw the scree plot (set False if it was already shown).

    Returns
    -------
    unrotated_loadings_df : pd.DataFrame
    rotated_loadings_df : pd.DataFrame
    communalities_df : pd.DataFrame
    unrotated_factor_scores_df : pd.DataFrame
    rotated_factor_scores_df : pd.DataFrame
    unrot_eigenvalues : np.ndarray
    rotated_variance_share : np.ndarray
    """

    # -------------------------------------------------------------------------
    # Step 1: Extract and Standardize Data
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 1] Extracting variables and standardizing data...")
        print(f"- Variables used: {variables}")
        print(f"- Number of observations: {data.shape[0]}")
        print(f"- Number of variables: {len(variables)}")

    data_subset = data[variables].copy()

    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_subset)

    if verbose:
        print("✔ Data standardization complete.")
        print(f"  Standardized matrix shape: {data_standardized.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Unrotated Factor Analysis
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 2] Performing *unrotated* factor analysis...")
        print(f"- Extracting {n_factors} factors")

    fa_unrot = FactorAnalyzer(n_factors=n_factors, rotation=None, method=method)
    fa_unrot.fit(data_standardized)

    unrotated_loadings = fa_unrot.loadings_
    unrotated_loadings_df = pd.DataFrame(
        unrotated_loadings,
        columns=[f"Component {i+1}" for i in range(n_factors)],
        index=variables
    )

    if verbose:
        print("✔ Unrotated loadings extracted.")
        print(f"  Loadings shape: {unrotated_loadings_df.shape}")

    # Factor scores (unrotated)
    unrotated_factor_scores = fa_unrot.transform(data_standardized)
    unrotated_factor_scores_df = pd.DataFrame(
        unrotated_factor_scores,
        columns=[f"Component {i+1}" for i in range(n_factors)]
    )

    if verbose:
        print("✔ Unrotated factor scores computed.")
        print(f"  Factor score shape: {unrotated_factor_scores_df.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Rotated Factor Analysis
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 3] Performing *rotated* factor analysis...")
        print(f"- Rotation method: {rotation}")

    fa_rot = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa_rot.fit(data_standardized)

    rotated_loadings = fa_rot.loadings_
    rotated_loadings_df = pd.DataFrame(
        rotated_loadings,
        columns=[f"Component {i+1}" for i in range(n_factors)],
        index=variables
    )

    if verbose:
        print("✔ Rotated loadings extracted.")
        print(f"  Loadings shape: {rotated_loadings_df.shape}")

    # Rotated factor scores
    rotated_factor_scores = fa_rot.transform(data_standardized)
    rotated_factor_scores_df = pd.DataFrame(
        rotated_factor_scores,
        columns=[f"Component {i+1}" for i in range(n_factors)]
    )

    if verbose:
        print("✔ Rotated factor scores computed.")
        print(f"  Factor score shape: {rotated_factor_scores_df.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Eigenvalues (Unrotated)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 4] Extracting eigenvalues from unrotated FA...")
    unrot_eigenvalues, _ = fa_unrot.get_eigenvalues()

    if verbose:
        print("✔ Eigenvalues extracted.")
        print(f"  Eigenvalues: {np.round(unrot_eigenvalues, 3)}")

    # -------------------------------------------------------------------------
    # Step 5: Variance Explained (Rotated)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 5] Calculating rotated variance explained...")

    rotated_variance_explained = np.sum(rotated_loadings**2, axis=0)
    rotated_variance_share = rotated_variance_explained / rotated_loadings.shape[0]

    if verbose:
        print("✔ Variance explained per rotated component:")
        for i, v in enumerate(rotated_variance_share, start=1):
            print(f"  - Component {i}: {v:.3f}")

    # -------------------------------------------------------------------------
    # Step 6: Communalities
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 6] Computing communalities...")

    communalities = np.sum(rotated_loadings**2, axis=1)
    communalities_df = pd.DataFrame(
        {"Variable": variables, "Communalities": communalities}
    )

    if verbose:
        print("✔ Communalities computed.")
        print(communalities_df)

    # -------------------------------------------------------------------------
    # Step 7: Plots
    # -------------------------------------------------------------------------
    if plot:
        if verbose:
            print("\n[Step 7] Generating diagnostic plots...")

        # ----- Scree Plot -----
        if show_scree:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(unrot_eigenvalues)+1), unrot_eigenvalues,
                     marker='o', linestyle='-', color='b', label='Eigenvalue')
            plt.axhline(y=1, color='gray', linestyle='--', label='Kaiser Criterion')
            plt.title("Scree Plot (Unrotated Solution)")
            plt.xlabel("Component Number")
            plt.ylabel("Eigenvalue")
            plt.grid(True)
            plt.legend()
            plt.show()

        # ----- Unrotated Heatmap -----
        plt.figure(figsize=(15, 12))
        sns.heatmap(unrotated_loadings_df, annot=True, cmap="coolwarm",
                    center=0, fmt=".2f", cbar_kws={'label': 'Loading'})
        plt.title(f"Unrotated Component Loadings ({n_factors} components)")
        plt.show()

        # ----- Rotated Heatmap -----
        plt.figure(figsize=(15, 12))
        sns.heatmap(rotated_loadings_df, annot=True, cmap="coolwarm",
                    center=0, fmt=".2f", cbar_kws={'label': 'Loading'})
        plt.title(f"Rotated Component Loadings ({rotation}, {n_factors} components)")
        plt.show()

    # -------------------------------------------------------------------------
    # Step 8: Return Results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[Step 8] Returning all outputs... Done.")

    return (
        unrotated_loadings_df,
        rotated_loadings_df,
        communalities_df,
        unrotated_factor_scores_df,
        rotated_factor_scores_df,
        unrot_eigenvalues,
        rotated_variance_share
    )


# =============================================================================
# Component-analysis workflow, in two layers:
#
#   computations and figures (no printing, no prompting; used by the web app and the CLI)
#     missing_summary        - missing values among the selected variables
#     overview_stats         - correlation matrix, eigenvalues, rules of thumb (independent of n)
#     correlation_figure, scree_figure
#     fit_solution           - one candidate solution: loadings, scores, variance, interpretation aids
#     unrotated_loadings_figure, rotated_loadings_figure
#     solution_summary_text  - the interpretation aids as text
#     component_candidates   - ranked candidates for the measure that represents one component
#
#   terminal workflow (prints the above and asks; used by pca_analysis.py and the notebook)
#     report_missing, pca_overview, pca_report, explore_pca, select_representatives
# =============================================================================

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def missing_summary(data, network_vars, ctrl_vars, id_col="id"):
    """
    Missing values among the selected variables. Returns a dict:
      counts  : Series of missing counts, only the variables with at least one (empty = fine)
      rows    : list of (id, [variables missing in that neighborhood])
      advice  : list of sentences saying how to deal with them
    The component analysis cannot run on missing values (the correlation matrix is undefined),
    so the callers stop when counts is not empty.
    """
    variables = list(network_vars) + list(ctrl_vars)
    counts = data[variables].isna().sum()
    counts = counts[counts > 0]
    rows, advice = [], []
    if not counts.empty:
        affected = data[variables].isna().any(axis=1)
        for _, r in data.loc[affected].iterrows():
            rows.append((r[id_col], [v for v in variables if pd.isna(r[v])]))
        if any(v in ctrl_vars for v in counts.index):
            advice.append("Control variables: fill or drop the values in the neighborhoods layer (the source "
                          "data), then rerun stage 1 with `python run_analysis.py --reuse`; the network measures "
                          "come back from the checkpoints in seconds, only the attributes are re-attached.")
        if any(v in network_vars for v in counts.index):
            advice.append("Network measures: these neighborhoods have no usable street network at that buffer "
                          "(no segment within the buffer, or fewer than two nodes). Check the streets layer "
                          "around them or remove them from the neighborhoods layer, then rerun stage 1.")
    return {"counts": counts, "rows": rows, "advice": advice}


def report_missing(data, network_vars, ctrl_vars, id_col="id"):
    """
    Prints the missing values among the selected variables (count per variable, then the
    affected neighborhoods with the variables they lack) and says how to deal with them.
    Returns the per-variable counts: empty when nothing is missing.
    """
    s = missing_summary(data, network_vars, ctrl_vars, id_col)
    counts = s["counts"]
    if counts.empty:
        print("No missing values in any of the selected variables.")
        return counts
    print(f"Missing values in {len(counts)} variable(s):")
    print(counts.to_string())
    print(f"\n{len(s['rows'])} neighborhood(s) affected:")
    for nid, missing in s["rows"]:
        print(f"  {id_col} {nid}: " + ", ".join(missing))
    print("\nThe component analysis cannot run on missing values: the correlation matrix is undefined.")
    for line in s["advice"]:
        print(f"- {line}")
    return counts


# ---------------------------------------------------------------------------
# Overview of one variable block (independent of the number of components)
# ---------------------------------------------------------------------------

def _eigenvalue_table(eigenvalues):
    pct = eigenvalues / eigenvalues.sum() * 100
    return pd.DataFrame({
        "Component": np.arange(1, len(eigenvalues) + 1),
        "Eigenvalue": eigenvalues,
        "% Variance": pct,
        "Cumulative %": np.cumsum(pct),
    }).set_index("Component")


def overview_stats(data, variables):
    """
    What can be judged before choosing the number of components: the correlation matrix of
    the standardized variables, all its eigenvalues (one per variable) with the share of
    variance each explains, and the component counts suggested by common rules of thumb.
    Returns a dict: variables, n_obs, correlation (DataFrame), eigenvalues (array),
    table (eigenvalue table indexed by component), rules ({"kaiser": n, "70%": n, "80%": n, "90%": n}).
    """
    variables = list(variables)
    X = StandardScaler().fit_transform(data[variables])
    corr = pd.DataFrame(np.corrcoef(X, rowvar=False), index=variables, columns=variables)
    eig = np.sort(np.linalg.eigvalsh(corr.values))[::-1]
    table = _eigenvalue_table(eig)
    cum = table["Cumulative %"].values
    rules = {"kaiser": int((eig > 1).sum())}
    for t in (70, 80, 90):
        rules[f"{t}%"] = int(np.searchsorted(cum, t) + 1)
    return {"variables": variables, "n_obs": int(X.shape[0]), "correlation": corr,
            "eigenvalues": eig, "table": table, "rules": rules}


def rules_of_thumb_text(rules):
    """One line, e.g. 'Kaiser (eigenvalue > 1) -> 6 components; 70% of variance -> 3; ...'."""
    return (f"Kaiser (eigenvalue > 1) -> {rules['kaiser']} components; "
            + "; ".join(f"{t}% of variance -> {rules[f'{t}%']}" for t in (70, 80, 90)))


def correlation_figure(stats, label=""):
    """Heatmap of the correlation matrix (values annotated up to 20 variables). Returns the figure."""
    corr = stats["correlation"]
    p = len(stats["variables"])
    title = f" ({label})" if label else ""
    size = min(0.45 * p + 4, 22)
    fig = plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=(p <= 20), fmt=".2f", square=True, cbar_kws={"label": "Pearson r"})
    plt.title(f"Correlation matrix{title}")
    fig.tight_layout()
    return fig


def scree_figure(stats, label=""):
    """Scree plot of all eigenvalues with the Kaiser line and the cumulative variance curve."""
    table = stats["table"]
    p = len(stats["variables"])
    title = f" ({label})" if label else ""
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(table.index, table["Eigenvalue"], marker="o", color="tab:blue", label="Eigenvalue")
    ax1.axhline(1, color="gray", linestyle="--", label="Kaiser criterion (eigenvalue = 1)")
    ax1.set_xlabel("Component number")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_xticks(table.index)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(table.index, table["Cumulative %"], marker="s", color="tab:orange",
             label="Cumulative % variance")
    ax2.set_ylabel("Cumulative % of variance")
    ax2.set_ylim(0, 105)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax2.set_title(f"Scree plot{title}: all {p} components")
    fig.tight_layout()
    return fig


def pca_overview(data, variables, label=""):
    """
    Terminal version of the overview: shows the correlation heatmap and the scree plot,
    prints the eigenvalue table and the rules of thumb. Returns the eigenvalue table.
    """
    stats = overview_stats(data, variables)
    title = f" ({label})" if label else ""
    print(f"\n=== PCA overview{title}: {len(stats['variables'])} variables, {stats['n_obs']} observations ===")
    correlation_figure(stats, label)
    plt.show()
    scree_figure(stats, label)
    plt.show()
    display(stats["table"].round(3))
    print(f"Rules of thumb: {rules_of_thumb_text(stats['rules'])}")
    return stats["table"]


# ---------------------------------------------------------------------------
# One candidate solution
# ---------------------------------------------------------------------------

def fit_solution(data, variables, n_factors, rotation="varimax", method="minres",
                 loading_threshold=0.5, communality_threshold=0.5):
    """
    Fits one candidate number of components and returns everything about it in a dict:
      n_factors, rotation, method
      unrotated_loadings, rotated_loadings   (DataFrames: variables x components)
      communalities                          (DataFrame: Variable, Communalities)
      unrotated_scores, rotated_scores       (DataFrames: observations x components)
      eigenvalues                            (array, all of them)
      variance_table                         (SS loadings, % Variance, Cumulative % per rotated component)
      salient          {component: Series of the loadings with |loading| >= loading_threshold, largest first}
      cross_loading    [variables salient on more than one component]
      not_salient      [variables salient on no component]
      low_communality  DataFrame of the variables with communality < communality_threshold
    Raises whatever the fit raises (e.g. a solution the extraction method cannot fit).
    """
    variables = list(variables)
    (unrot_load, rot_load, communalities, unrot_scores, rot_scores,
     eigenvalues, _) = factor_loading_matrices(
        data, variables, n_factors, rotation=rotation, method=method, plot=False, verbose=False)

    ss = (rot_load ** 2).sum(axis=0)
    var_tab = pd.DataFrame({"SS loadings": ss, "% Variance": ss / len(variables) * 100})
    var_tab["Cumulative %"] = var_tab["% Variance"].cumsum()

    salient = {}
    for comp in rot_load.columns:
        s = rot_load[comp]
        salient[comp] = s[s.abs() >= loading_threshold].sort_values(key=np.abs, ascending=False)
    is_salient = rot_load.abs() >= loading_threshold
    cross = is_salient.sum(axis=1)
    weak = rot_load.abs().max(axis=1)
    low = communalities[communalities["Communalities"] < communality_threshold]

    return {
        "n_factors": n_factors, "rotation": rotation, "method": method,
        "unrotated_loadings": unrot_load, "rotated_loadings": rot_load,
        "communalities": communalities,
        "unrotated_scores": unrot_scores, "rotated_scores": rot_scores,
        "eigenvalues": eigenvalues, "variance_table": var_tab,
        "loading_threshold": loading_threshold, "communality_threshold": communality_threshold,
        "salient": salient,
        "cross_loading": list(cross[cross > 1].index),
        "not_salient": list(weak[weak < loading_threshold].index),
        "low_communality": low,
    }


def _loadings_heatmap(loadings, title):
    fig = plt.figure(figsize=(15, 12))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, fmt=".2f", cbar_kws={"label": "Loading"})
    plt.title(title)
    return fig


def unrotated_loadings_figure(report):
    return _loadings_heatmap(report["unrotated_loadings"],
                             f"Unrotated Component Loadings ({report['n_factors']} components)")


def rotated_loadings_figure(report):
    return _loadings_heatmap(report["rotated_loadings"],
                             f"Rotated Component Loadings ({report['rotation']}, {report['n_factors']} components)")


def solution_summary_text(report):
    """The interpretation aids of a solution as text (one line per item), as the terminal prints them."""
    lines = [f"Salient loadings (|loading| >= {report['loading_threshold']}) per component:"]
    for comp, s in report["salient"].items():
        lines.append(f"  {comp}: " + (", ".join(f"{v} ({s[v]:+.2f})" for v in s.index) or "none"))
    lines.append("")
    lines.append("Cross-loading variables (salient on more than one component): "
                 + (", ".join(report["cross_loading"]) or "none"))
    lines.append("Variables not salient on any component: " + (", ".join(report["not_salient"]) or "none"))
    low = report["low_communality"]
    lines.append(f"Low communality (< {report['communality_threshold']}): "
                 + (", ".join(f"{r.Variable} ({r.Communalities:.2f})" for r in low.itertuples()) if len(low) else "none"))
    return "\n".join(lines)


def pca_report(data, variables, n_factors, rotation="varimax", method="minres",
               loading_threshold=0.5, communality_threshold=0.5):
    """
    Terminal version of the report for one candidate number of components: shows the
    loadings heatmaps, prints the variance explained, the matrices, the communalities and
    the interpretation aids. Returns the dict from fit_solution().
    """
    print(f"\n{'=' * 78}\n REPORT: {n_factors} component(s)   rotation = {rotation}   method = {method}\n{'=' * 78}")
    report = fit_solution(data, variables, n_factors, rotation=rotation, method=method,
                          loading_threshold=loading_threshold, communality_threshold=communality_threshold)
    unrotated_loadings_figure(report)
    plt.show()
    rotated_loadings_figure(report)
    plt.show()

    var_tab = report["variance_table"]
    print("\nVariance explained by the rotated components:")
    display(var_tab.round(3))
    print(f"Total variance retained by {n_factors} component(s): {var_tab['Cumulative %'].iloc[-1]:.1f}%")

    print("\nUnrotated loadings:")
    display(report["unrotated_loadings"].round(3))
    print(f"\nRotated loadings ({rotation}):")
    display(report["rotated_loadings"].round(3))
    print("\nCommunalities (share of each variable's variance captured by the solution):")
    display(report["communalities"].set_index("Variable").round(3).T)

    print()
    print(solution_summary_text(report))
    return report


def explore_pca(data, variables, label, rotation="varimax", method="minres"):
    """
    Interactive workflow for choosing the number of components of one variable block:
      1. overview (correlation heatmap, scree plot, eigenvalue table), independent of n;
      2. loop: type a candidate n to see its full report; repeat with other values freely;
      3. prompt for the final n to retain (Enter keeps the last one inspected).
    Returns (n_final, report), where report is the dict from pca_report().
    """
    p = len(variables)
    pca_overview(data, variables, label)
    reports, last = {}, None

    def ask(prompt):
        return input(prompt).strip()

    # ----- Stage 2: inspect candidate solutions -----
    while True:
        ans = ask(f"\n[{label}] Number of components to inspect (1-{p}), or press Enter to stop exploring: ")
        if ans == "":
            if last is None:
                print("  -> inspect at least one solution first")
                continue
            break
        if not ans.isdigit() or not 1 <= int(ans) <= p:
            print(f"  -> enter an integer between 1 and {p}")
            continue
        n = int(ans)
        try:
            reports[n] = pca_report(data, variables, n, rotation=rotation, method=method)
            last = n
        except Exception as e:
            print(f"  -> could not fit a {n}-component solution: {e}")

    # ----- Stage 3: final choice -----
    while True:
        ans = ask(f"[{label}] Final number of components to retain [{last}]: ")
        if ans == "":
            n_final = last
            break
        if ans.isdigit() and 1 <= int(ans) <= p:
            n_final = int(ans)
            if n_final not in reports:
                try:
                    reports[n_final] = pca_report(data, variables, n_final, rotation=rotation, method=method)
                except Exception as e:
                    print(f"  -> could not fit a {n_final}-component solution: {e}")
                    continue
            break
        print(f"  -> enter an integer between 1 and {p}")

    print(f"\n[{label}] Retaining {n_final} component(s).")
    return n_final, reports[n_final]


# ---------------------------------------------------------------------------
# Representative measures
# ---------------------------------------------------------------------------

def component_candidates(report, component, exclude=(), top_k=5):
    """
    The measures that could stand for one rotated component, ranked by absolute loading,
    leaving out those in `exclude` (already taken by another component). Each candidate is a
    dict: measure, loading (on this component), max_elsewhere (its largest absolute loading
    on any other component, None if there is none) and where (that component).
    """
    rot = report["rotated_loadings"]
    ranked = rot[component].abs().sort_values(ascending=False)
    out = []
    for v in ranked.index:
        if v in exclude:
            continue
        others = rot.loc[v].drop(component).abs()
        out.append({"measure": v, "loading": float(rot.loc[v, component]),
                    "max_elsewhere": float(others.max()) if len(others) else None,
                    "where": others.idxmax() if len(others) else None})
        if len(out) == top_k:
            break
    return out


def candidate_text(c):
    """One line describing a candidate, e.g. 'int_den075   loading +0.96   max elsewhere 0.12 on Component 2'."""
    cross = (f"max elsewhere {c['max_elsewhere']:.2f} on {c['where']}" if c["max_elsewhere"] is not None
             else "no other component")
    return f"{c['measure']:<12s} loading {c['loading']:+.2f}   {cross}"


def select_representatives(report, label="", top_k=5):
    """
    Interactive choice of one original measure to stand for each rotated component
    (the paper's Table 2 step): the components themselves are not entered in the
    regression; the chosen measures are, after a log(x + 1) transformation.
    For each component the top-loading measures are listed with their loading on
    that component and their largest loading on any other component (cross-loading).
    Type a number or a measure name, or press Enter to take the highest loader.
    A measure cannot represent two components. Returns the chosen names in
    component order.
    """
    rot = report["rotated_loadings"]
    chosen = []
    title = f" ({label})" if label else ""
    print(f"\n=== Representative measures{title}: one original measure per component ===")
    print("Rule of thumb: the highest absolute loading; when two are close, prefer the one with the "
          "smaller cross-loading and the clearer theoretical meaning.")
    for comp in rot.columns:
        candidates = component_candidates(report, comp, exclude=chosen, top_k=top_k)
        names = [c["measure"] for c in candidates]
        print(f"\n{comp}:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {candidate_text(c)}")
        while True:
            ans = input(f"  Measure for {comp} [{names[0]}]: ").strip()
            if ans == "":
                pick = names[0]
                break
            if ans.isdigit() and 1 <= int(ans) <= len(names):
                pick = names[int(ans) - 1]
                break
            if ans in rot.index and ans not in chosen:
                pick = ans
                break
            print("  -> enter a number from the list, an unused measure name, or Enter for the first")
        chosen.append(pick)
        print(f"  -> {comp}: {pick}")
    print("\nSelected representatives: " + ", ".join(chosen))
    display(rot.loc[chosen].round(3))
    return chosen

# =============================================================================
# Keeping the figures: FigureSaver writes every figure that plt.show() is about to
# display to a folder, then displays it as usual (or closes it). Used by both
# pca_analysis.py and pca_analysis.ipynb so the diagnostics stay on disk.
# =============================================================================

_ORIGINAL_SHOW = plt.show   # captured once at import time, so install() can be called repeatedly


class FigureSaver:
    """
    Saves every figure that plt.show() is about to display as a numbered PNG in
    out_dir, named after the current stage and the figure title, and then either
    displays it as usual (show=True: a window in a script, inline in a notebook)
    or closes it without displaying (show=False).

        figures = FigureSaver("../output/pca_diagnostics").install()
        figures.stage = "network"      # prefix of the files saved from now on
    """

    def __init__(self, out_dir, show=True):
        self.out_dir = Path(out_dir)
        self.show = show
        self.stage = "figure"
        self._counter = itertools.count(1)

    @staticmethod
    def slug(text):
        return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()[:60]

    def install(self):
        """Route plt.show() through this saver (safe to call more than once)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        saver = self

        @functools.wraps(_ORIGINAL_SHOW)
        def show(*args, **kwargs):          # a plain function: matplotlib sets attributes on plt.show
            saver._show_and_save(*args, **kwargs)

        plt.show = show
        return self

    def save_open_figures(self):
        paths = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
            xlabels = [ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()]
            label = (titles or xlabels or ["figure"])[0]
            path = self.out_dir / f"{next(self._counter):03d}_{self.slug(self.stage)}_{self.slug(label)}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            paths.append(path)
        return paths

    def _show_and_save(self, *args, **kwargs):
        for path in self.save_open_figures():
            print(f"  figure saved: {path}")
        if self.show:
            _ORIGINAL_SHOW(*args, **kwargs)   # in a window this returns once the window is closed
        plt.close("all")                      # never keep figures around to be saved twice
