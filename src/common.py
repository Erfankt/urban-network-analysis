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
# Interactive PCA workflow (used by pca_analysis.ipynb)
#   1. pca_overview  - what can be judged before choosing n (independent of n)
#   2. pca_report    - the full diagnostic report for one candidate n
#   3. explore_pca   - the interactive loop: overview -> inspect n's -> final n
# =============================================================================

def _eigenvalue_table(eigenvalues):
    pct = eigenvalues / eigenvalues.sum() * 100
    return pd.DataFrame({
        "Component": np.arange(1, len(eigenvalues) + 1),
        "Eigenvalue": eigenvalues,
        "% Variance": pct,
        "Cumulative %": np.cumsum(pct),
    }).set_index("Component")


def pca_overview(data, variables, label=""):
    """
    Stage 1: the diagnostics that do not depend on the number of components.
    Shows the correlation heatmap of the variables and the scree plot of all
    eigenvalues (one per variable) with the Kaiser line and the cumulative
    variance curve, prints the eigenvalue table and the component counts
    suggested by common rules of thumb. Returns the eigenvalue table.
    """
    X = StandardScaler().fit_transform(data[variables])
    p = len(variables)
    corr = np.corrcoef(X, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    table = _eigenvalue_table(eig)
    title = f" ({label})" if label else ""

    print(f"\n=== PCA overview{title}: {p} variables, {X.shape[0]} observations ===")

    # ----- Correlation heatmap -----
    size = min(0.45 * p + 4, 22)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(pd.DataFrame(corr, index=variables, columns=variables),
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=(p <= 20), fmt=".2f", square=True,
                cbar_kws={"label": "Pearson r"})
    plt.title(f"Correlation matrix{title}")
    plt.tight_layout()
    plt.show()

    # ----- Scree plot with cumulative variance -----
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
    plt.title(f"Scree plot{title}: all {p} components")
    plt.tight_layout()
    plt.show()

    display(table.round(3))
    kaiser = int((eig > 1).sum())
    cum = table["Cumulative %"].values
    rules = "; ".join(f"{t}% of variance -> {int(np.searchsorted(cum, t) + 1)}" for t in (70, 80, 90))
    print(f"Rules of thumb: Kaiser (eigenvalue > 1) -> {kaiser} components; {rules}")
    return table


def pca_report(data, variables, n_factors, rotation="varimax", method="minres",
               loading_threshold=0.5, communality_threshold=0.5):
    """
    Stage 2: the full diagnostic report for one candidate number of components:
    unrotated and rotated loadings (tables and heatmaps), communalities,
    variance explained per rotated component, and interpretation aids
    (salient loadings per component, cross-loading variables, variables not
    captured by any component, low communalities).
    Returns a dict holding every matrix so the chosen solution can be reused.
    """
    print(f"\n{'=' * 78}\n REPORT: {n_factors} component(s)   rotation = {rotation}   method = {method}\n{'=' * 78}")
    (unrot_load, rot_load, communalities, unrot_scores, rot_scores,
     eigenvalues, rot_var_share) = factor_loading_matrices(
        data, variables, n_factors, rotation=rotation, method=method,
        plot=True, verbose=False, show_scree=False)

    # ----- Variance explained by the rotated components -----
    ss = (rot_load ** 2).sum(axis=0)
    var_tab = pd.DataFrame({"SS loadings": ss, "% Variance": ss / len(variables) * 100})
    var_tab["Cumulative %"] = var_tab["% Variance"].cumsum()
    print("\nVariance explained by the rotated components:")
    display(var_tab.round(3))
    print(f"Total variance retained by {n_factors} component(s): {var_tab['Cumulative %'].iloc[-1]:.1f}%")

    # ----- Matrices -----
    print("\nUnrotated loadings:")
    display(unrot_load.round(3))
    print(f"\nRotated loadings ({rotation}):")
    display(rot_load.round(3))
    print("\nCommunalities (share of each variable's variance captured by the solution):")
    display(communalities.set_index("Variable").round(3).T)

    # ----- Interpretation aids -----
    print(f"\nSalient loadings (|loading| >= {loading_threshold}) per component:")
    for comp in rot_load.columns:
        s = rot_load[comp]
        s = s[s.abs() >= loading_threshold].sort_values(key=np.abs, ascending=False)
        print(f"  {comp}: " + (", ".join(f"{v} ({s[v]:+.2f})" for v in s.index) or "none"))

    salient = rot_load.abs() >= loading_threshold
    cross = salient.sum(axis=1)
    cross = cross[cross > 1]
    print("\nCross-loading variables (salient on more than one component): "
          + (", ".join(cross.index) if len(cross) else "none"))
    weak = rot_load.abs().max(axis=1)
    weak = weak[weak < loading_threshold]
    print("Variables not salient on any component: "
          + (", ".join(weak.index) if len(weak) else "none"))
    low = communalities[communalities["Communalities"] < communality_threshold]
    print(f"Low communality (< {communality_threshold}): "
          + (", ".join(f"{r.Variable} ({r.Communalities:.2f})" for r in low.itertuples()) if len(low) else "none"))

    return {
        "n_factors": n_factors, "rotation": rotation, "method": method,
        "unrotated_loadings": unrot_load, "rotated_loadings": rot_load,
        "communalities": communalities,
        "unrotated_scores": unrot_scores, "rotated_scores": rot_scores,
        "eigenvalues": eigenvalues, "variance_table": var_tab,
    }


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
        ranked = rot[comp].abs().sort_values(ascending=False)
        candidates = [v for v in ranked.index if v not in chosen][:top_k]
        print(f"\n{comp}:")
        for i, v in enumerate(candidates, 1):
            others = rot.loc[v].drop(comp).abs()
            cross = f"max elsewhere {others.max():.2f} on {others.idxmax()}" if len(others) else "no other component"
            print(f"  {i}. {v:<12s} loading {rot.loc[v, comp]:+.2f}   {cross}")
        while True:
            ans = input(f"  Measure for {comp} [{candidates[0]}]: ").strip()
            if ans == "":
                pick = candidates[0]
                break
            if ans.isdigit() and 1 <= int(ans) <= len(candidates):
                pick = candidates[int(ans) - 1]
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
