import matplotlib.pyplot as plt,seaborn as sns
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer





def factor_loading_matrices(
    data,
    variables,
    n_factors,
    rotation='varimax',
    plot=True,
    verbose=True
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

    fa_unrot = FactorAnalyzer(n_factors=n_factors, rotation=None)
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

    fa_rot = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
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
        plt.title("Unrotated Component Loadings")
        plt.show()

        # ----- Rotated Heatmap -----
        plt.figure(figsize=(15, 12))
        sns.heatmap(rotated_loadings_df, annot=True, cmap="coolwarm",
                    center=0, fmt=".2f", cbar_kws={'label': 'Loading'})
        plt.title(f"Rotated Component Loadings ({rotation})")
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