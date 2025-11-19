import datetime
import time
import os, re, time
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import json
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix


def _round4(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4)


def _round4_list(xs):
    if xs is None:
        return None
    # if string, try to parse
    if isinstance(xs, str):
        try:
            xs = json.loads(xs)
        except Exception:
            return xs  # keep as-is if it's not valid JSON
    try:
        return [round(float(v), 4) for v in xs]
    except Exception:
        return xs  # fallback if not iterable/castable


def _fold_str(xs_rounded):
    if xs_rounded is None:
        return None
    # stable JSON string for comparison/storage
    return json.dumps(xs_rounded, separators=(",", ":"))


def log_results_excel(
    file_path, model_desc, accuracy, std=None, fold_scores=None, runtime=None
):
    """
    Save model evaluation results to an Excel log file.

    Behavior:
    - If an entry with the same description and identical metrics exists, do nothing.
    - If description exists but metrics differ, replace the row.
    - Otherwise, append as new row.
    - Always sorts by mean accuracy (descending, better on top).

    Parameters
    ----------
    file_path : str
        Path to Excel log file.
    model_desc : str
        Description of the model + feature set.
    accuracy : float
        Mean cross-validation accuracy.
    std : float, optional
        Standard deviation of fold accuracies.
    fold_scores : list[float], optional
        List of fold accuracies.
    runtime : float, optional
        Training runtime in seconds.

    Returns
    -------
    None
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mean_acc_4 = _round4(accuracy)
    std_4 = _round4(std)
    folds_4 = _round4_list(fold_scores)
    folds_str = _fold_str(folds_4)

    new_row = {
        "timestamp": timestamp,
        "description": model_desc,
        "mean_accuracy": mean_acc_4,
        "std_dev": std_4,
        "fold_scores": folds_str,
        "runtime_sec": (
            float(runtime) if runtime is not None else None
        ),  # saved but not compared
    }

    cols = [
        "timestamp",
        "description",
        "mean_accuracy",
        "std_dev",
        "fold_scores",
        "runtime_sec",
    ]

    # load or init file
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)

    # check for existing description
    mask = df["description"] == model_desc
    if mask.any():
        existing = df.loc[mask].iloc[-1]

        # normalize existing values to the same 4dp representation
        ex_mean = _round4(existing.get("mean_accuracy"))
        ex_std = _round4(existing.get("std_dev"))
        ex_folds_str = _fold_str(_round4_list(existing.get("fold_scores")))

        same = (
            ex_mean == mean_acc_4
            and ex_std == std_4
            and ex_folds_str == folds_str
            # runtime_sec intentionally ignored
        )

        if same:
            print(f"⚠️ No changes for '{model_desc}', not updating log.")
            return
        else:
            # remove all old rows for this description, then append the new one
            df = df.loc[~mask].copy()
            df = pd.concat([df, pd.DataFrame([new_row])[cols]], ignore_index=True)
            print(f"🔄 Updated results for '{model_desc}'.")
    else:
        df = pd.concat([df, pd.DataFrame([new_row])[cols]], ignore_index=True)
        print(f"➕ Added new results for '{model_desc}'.")

    df = df.sort_values(
        by="mean_accuracy", ascending=False, na_position="last"
    ).reset_index(drop=True)
    df.to_excel(file_path, index=False)
    print(f"✅ Results saved to {file_path}")


def create_folds(input_data, output_file, text_columns, n_splits=5, random_seed=42):
    """
    Create stratified K-Folds from a dataset and save to CSV.

    - Keeps ALL original columns.
    - Adds a 'document' column: either a single text column or concatenation of multiple.
    - Adds a 'fold' column with values 1..n_splits.

    Parameters
    ----------
    input_data : str | pd.DataFrame
        Input dataset (CSV path or DataFrame).
    output_file : str
        Path to save folds CSV.
    text_columns : str | list[str]
        One or more text columns to concatenate into 'document'.
    n_splits : int, default=5
        Number of folds.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    None
    """

    if isinstance(input_data, pd.DataFrame):
        data = input_data.copy()
    else:
        print(f"Loading data from '{input_data}'...")
        data = pd.read_csv(input_data, sep=None, engine="python")

    # fill NaNs on the text columns and cast to string
    for c in text_columns:
        if c not in data.columns:
            raise KeyError(
                f"Column '{c}' not found in {input_data}. Available: {list(data.columns)}"
            )
        data[c] = data[c].fillna("").astype(str)

    # create/overwrite combined text column 'document'
    if len(text_columns) == 1:
        data["document"] = data[text_columns[0]]
    else:
        data["document"] = data[text_columns].apply(" ".join, axis=1)

    if "chef_id" not in data.columns:
        raise KeyError("Required target column 'chef_id' not found in the input file.")

    X = data["document"]
    y = data["chef_id"]

    # build folds on the full (original-order) DataFrame
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds_df = data.copy().reset_index(drop=True)  # <-- keep ALL columns
    folds_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(X, y), start=1):
        folds_df.loc[val_idx, "fold"] = fold

    # --- sanity check ---
    print("\n" + "=" * 50)
    print("## Chef ID Distribution Check")
    overall_counts = y.value_counts()
    overall_percent = y.value_counts(normalize=True).mul(100).round(2)
    print(f"\n=== Original Dataset === (Size: {len(y)})")
    print(pd.DataFrame({"Count": overall_counts, "Percentage": overall_percent}))
    for fold in range(1, n_splits + 1):
        subset = folds_df.loc[folds_df["fold"] == fold, "chef_id"]
        percentages = subset.value_counts(normalize=True).mul(100).round(2)
        print(f"\n=== Fold {fold} === (Size: {len(subset)})")
        print(pd.DataFrame({"Count": subset.value_counts(), "Percentage": percentages}))
    print("=" * 50)

    folds_df.to_csv(output_file, index=False)  # use sep=';' if you prefer
    print(f"\nFolds saved to '{output_file}' (Total: {len(folds_df)} rows)")


def run_kfold_experiment(
    folds_file,
    model_cls,
    text_columns="document",  # str | list[str] | dict[str, dict]
    categorical_columns=None,  # str | list[str]
    numerical_columns=None,  # str | list[str]
    model_kwargs=None,
    model_desc=None,
    log_path="results/log.xlsx",
    random_seed=42,
    log_results=False,
    return_last=False,
    save_model=False,
):
    """
    Run K-Fold cross-validation for a given model class on text, categorical, and numerical features.

    Features are processed as follows:
    - Text: TF-IDF vectorizer (per column).
    - Categorical: MultiLabelBinarizer (parsed from list-like strings).
    - Numerical: Scaled (StandardScaler or MinMaxScaler depending on model).

    Parameters
    ----------
    folds_file : str
        Path to pre-split folds CSV file (with 'fold' and 'chef_id' columns).
    model_cls : type
        Model class (e.g., sklearn SVC, LogisticRegression).
    text_columns : str | list[str] | dict[str, dict], default="document"
        Text feature(s). If dict, per-column TF-IDF config.
    categorical_columns : str | list[str], optional
        Categorical feature(s).
    numerical_columns : str | list[str], optional
        Numerical feature(s).
    model_kwargs : dict, optional
        Extra arguments passed to model constructor.
    model_desc : str, optional
        Description for logging.
    log_path : str, default="results/log.xlsx"
        Path to log file for results.
    random_seed : int, default=42
        Random seed for reproducibility.
    log_results : bool, default=False
        Whether to log results in Excel.
    return_last : bool, default=False
        If True, returns last fold's trained model and validation data.
    save_model : bool, default=False
        If True, saves the trained model after the final fold has been processed.


    Returns
    -------
    dict
        Summary with mean accuracy, std, runtime, fold scores, and optionally last fold info.
    """

    model_kwargs = model_kwargs or {}
    try:
        if "random_state" in model_cls().get_params():
            model_kwargs.setdefault("random_state", random_seed)
    except Exception:
        pass

    folds_df = pd.read_csv(folds_file)

    # ---- helpers ----
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        try:
            import numpy as _np

            if isinstance(x, _np.ndarray):
                return list(x)
        except Exception:
            pass
        try:
            if isinstance(x, pd.Index):
                return list(x)
        except Exception:
            pass
        return [x]  # wrap singletons (incl. plain strings)

    def parse_tags(tag_string):
        import re, json, ast

        def _clean(x):
            return str(x).strip().strip('"').strip("'")

        s = "" if tag_string is None else str(tag_string).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return []

        # 1) Safe Python literal (handles single/double quotes)
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [_clean(x) for x in v if _clean(x)]
            if isinstance(v, dict):
                return [_clean(k) for k in v.keys() if _clean(k)]
        except Exception:
            pass

        # 2) JSON list
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [_clean(x) for x in v if _clean(x)]
        except Exception:
            pass

        # 3) Quoted tokens
        m = re.findall(r"""(['"])(.*?)\1""", s)
        if m:
            return [_clean(x[1]) for x in m if _clean(x[1])]

        # 4) Split on separators
        core = s.strip("[]")
        return [_clean(t) for t in re.split(r"[|;,]", core) if _clean(t)]

    # --- normalize text_columns ---
    if text_columns is None:
        text_cols = []
        tfidf_cfg = {}
    elif isinstance(text_columns, str):
        text_cols = [text_columns]
        tfidf_cfg = {text_columns: {}}
    elif isinstance(text_columns, list):
        text_cols = list(text_columns)
        tfidf_cfg = {c: {} for c in text_cols}
    elif isinstance(text_columns, dict):
        text_cols = list(text_columns.keys())
        tfidf_cfg = text_columns
    else:
        raise ValueError("text_columns must be str | list[str] | dict[str, dict].")

    # --- normalize categorical/numerical ---
    categorical_columns = _as_list(categorical_columns)
    numerical_columns = _as_list(numerical_columns)

    # quick required-columns check (helps catch typos early)
    required = (
        set(text_cols)
        | set(categorical_columns)
        | set(numerical_columns)
        | {"chef_id", "fold"}
    )
    missing = [c for c in required if c not in folds_df.columns]
    if missing:
        raise KeyError(f"Missing columns in '{folds_file}': {missing}")

    fold_ids = sorted(folds_df["fold"].unique())
    fold_scores = []
    start_time = time.time()
    last = None

    for fold in fold_ids:
        train_df = folds_df[folds_df["fold"] != fold].reset_index(drop=True)
        val_df = folds_df[folds_df["fold"] == fold].reset_index(drop=True)

        # --- TEXT (per column TF-IDF, fit on train only) ---
        text_mats_train, text_mats_val = [], []
        for col in text_cols:
            cfg = {
                "lowercase": True,
                "token_pattern": r"(?u)\b[a-z]{2,}\b",
                **(tfidf_cfg.get(col) or {}),
            }
            vec = TfidfVectorizer(**cfg)
            Xtr = vec.fit_transform(train_df[col].fillna(""))
            Xva = vec.transform(val_df[col].fillna(""))
            text_mats_train.append(Xtr)
            text_mats_val.append(Xva)

        # --- CATEGORICAL (each is list-like; MLB fit on train only) ---
        cat_mats_train, cat_mats_val = [], []
        for col in categorical_columns:
            mlb = MultiLabelBinarizer()
            Xtr = mlb.fit_transform(train_df[col].apply(parse_tags))
            Xva = mlb.transform(val_df[col].apply(parse_tags))
            cat_mats_train.append(csr_matrix(Xtr))
            cat_mats_val.append(csr_matrix(Xva))

        # --- NUMERICAL (scaled per model; fit on train only) ---
        if numerical_columns:
            Xtr_num_raw = train_df[numerical_columns].astype(float).to_numpy()
            Xva_num_raw = val_df[numerical_columns].astype(float).to_numpy()
            if model_cls.__name__ in ["MultinomialNB", "ComplementNB"]:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            Xtr_num = csr_matrix(scaler.fit_transform(Xtr_num_raw))
            Xva_num = csr_matrix(scaler.transform(Xva_num_raw))
        else:
            Xtr_num = csr_matrix((len(train_df), 0))
            Xva_num = csr_matrix((len(val_df), 0))

        # --- STACK ALL FEATURES ---
        X_train_features = hstack(
            text_mats_train + cat_mats_train + [Xtr_num], format="csr"
        )
        X_val_features = hstack(text_mats_val + cat_mats_val + [Xva_num], format="csr")

        y_train = train_df["chef_id"]
        y_val = val_df["chef_id"]

        # --- FIT & PREDICT (handle LightGBM feature-names warning cleanly) ---
        model = model_cls(**model_kwargs)
        is_lgbm = model.__class__.__name__ == "LGBMClassifier"
        if is_lgbm:
            # Ensure consistent feature names at fit and predict time
            from scipy import sparse as sp

            n_feats = X_train_features.shape[1]
            cols = [f"f{i}" for i in range(n_feats)]
            if sp.issparse(X_train_features):
                Xtr_df = pd.DataFrame.sparse.from_spmatrix(
                    X_train_features, columns=cols
                )
                Xva_df = pd.DataFrame.sparse.from_spmatrix(X_val_features, columns=cols)
            else:
                Xtr_df = pd.DataFrame(X_train_features, columns=cols)
                Xva_df = pd.DataFrame(X_val_features, columns=cols)
            model.fit(Xtr_df, y_train)
            y_pred = model.predict(Xva_df)
        else:
            model.fit(X_train_features, y_train)
            y_pred = model.predict(X_val_features)

        acc = accuracy_score(y_val, y_pred)
        fold_scores.append(acc)

        if return_last and fold == fold_ids[0]:
            last = {
                "model": model,
                "X_val": X_val_features,
                "y_val": y_val,
                "y_pred": y_pred,
                "val_df": val_df.copy(),
                "feature_names": cols if is_lgbm else None,
            }
    
    # save the model after the final fold (if `save_model` is True)
    if save_model:
        model_filename = f"models/{model_desc}.joblib"
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_desc}.joblib")

    runtime = time.time() - start_time
    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

    print("=" * 50)
    print(f"Model: {model_cls.__name__}")
    print(f"Cross-Validation (using {folds_file})")
    for f, s in zip(fold_ids, fold_scores):
        print(f"  Fold {f}: {s:.4f}")
    print(f"Mean accuracy: {mean_acc:.4f}  |  Std: {std_acc:.4f}")
    print(f"Total runtime: {runtime:.2f} seconds")
    print("=" * 50)

    if log_results:
        # assumes you have the 4-decimal rounding in log_results_excel on your side
        log_results_excel(
            file_path=log_path,
            model_desc=model_desc or f"{model_cls.__name__} + multiTEXT + CATS + NUM",
            accuracy=mean_acc,
            std=std_acc,
            fold_scores=[round(v, 4) for v in fold_scores],
            runtime=runtime,
        )

    results = {
        "Model": model_cls.__name__,
        "Model_Desc": model_desc or f"{model_cls.__name__} + multiTEXT + CATS + NUM",
        "Folds": list(fold_ids),
        "Fold_Scores": [float(s) for s in fold_scores],
        "Accuracy_Mean": mean_acc,
        "Accuracy_Std": std_acc,
        "Runtime_Sec": float(runtime),
        "Num_Folds": int(len(fold_ids)),
    }
    if return_last:
        results["Last_Run"] = last
    return results


def run_configs(
    models,
    folds_path,
    *,
    tfidf_configs=None,  # list[dict] with TF-IDF params + "description"
    feature_combinations=None,  # default becomes [("Text",)]
    text_features="document",  # str | list[str] default same as create_folds: "document"
    categorical_features=None,  # str | list[str]
    numerical_features=None,  # str | list[str]
    what_data=None,
    random_seed=42,
    log_results=False,
):
    """
    Run multiple models across combinations of features and TF-IDF configs.

    Parameters
    ----------
    models : list[tuple]
        List of (model_cls, model_kwargs, description).
    folds_path : str
        Path to folds CSV file.
    tfidf_configs : list[dict], optional
        List of TF-IDF configs, each with "description" and params.
    feature_combinations : list[tuple], optional
        List of feature set combinations (e.g., ("Text", "Numerical")).
        Defaults to [("Text",)].
    text_features : str | list[str], default="document"
        Text column(s).
    categorical_features : str | list[str], optional
        Categorical column(s).
    numerical_features : str | list[str], optional
        Numerical column(s).
    what_data : str, default="None", optional
        Tag appended to the model description.
        The experiment description becomes:
            "{description} ({what_data} data) | {feature_combo}".
    random_seed : int, default=42
        Random seed for reproducibility.
    log_results : bool, default=False
        Whether to log to Excel.

    Returns
    -------
    pd.DataFrame
        Results sorted by accuracy (descending).
    """

    # helpers
    def _infer_what_data(path: str) -> str:
        """Extract tag between last '_' and final '.' of the filename."""
        fname = os.path.basename(path)
        stem, _ = os.path.splitext(fname)  # e.g., folds_preprocessed
        if "_" in stem:
            return stem.rsplit("_", 1)[-1] or "unknown"
        return "unknown"

    # normalize list-like args
    def _as_list(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple, set)):
            return list(x)
        return [x]

    # map TF-IDF cfg to 1..N text columns
    def _make_text_arg(cols, tfidf_cfg):
        if isinstance(cols, str):
            return {cols: tfidf_cfg}
        elif isinstance(cols, list):
            return {c: tfidf_cfg for c in cols}
        else:
            raise ValueError("text_features must be str or list[str]")

    def _add_row(rows, model_desc, summary):
        rows.append(
            {
                "Algorithm": model_desc,
                "Accuracy": summary["Accuracy_Mean"],
                "Training_Time_s": summary["Runtime_Sec"],
            }
        )

    # safe default for feature_combinations
    if feature_combinations is None:
        feature_combinations = [("Text",)]

    if what_data is None:
        inferred_tag = _infer_what_data(folds_path)
    else:
        inferred_tag = what_data

    categorical_features = _as_list(categorical_features)
    numerical_features = _as_list(numerical_features)
    tfidf_configs = tfidf_configs or [{"description": "(default TF-IDF)"}]

    results = []

    for cls, kwargs, desc in models:
        for combo in feature_combinations:
            combo_name = " + ".join(combo)

            cats_arg = (
                categorical_features
                if ("Categorical" in combo and categorical_features)
                else None
            )
            nums_arg = (
                numerical_features
                if ("Numerical" in combo and numerical_features)
                else None
            )

            if "Text" not in combo:
                model_desc = f"{desc} ({inferred_tag} data) | {combo_name}"
                summary = run_kfold_experiment(
                    folds_file=folds_path,
                    model_cls=cls,
                    text_columns=None,
                    categorical_columns=cats_arg,
                    numerical_columns=nums_arg,
                    model_kwargs=kwargs,
                    model_desc=model_desc,
                    random_seed=random_seed,
                    log_results=log_results,
                )
                _add_row(results, model_desc, summary)
            else:
                for cfg in tfidf_configs:
                    model_desc = f"{desc} ({inferred_tag} data) | {combo_name} | {cfg['description']}"
                    tfidf_cfg = {k: v for k, v in cfg.items() if k != "description"}
                    summary = run_kfold_experiment(
                        folds_file=folds_path,
                        model_cls=cls,
                        text_columns=(
                            _make_text_arg(
                                (
                                    _as_list(text_features)
                                    if isinstance(text_features, str)
                                    else text_features
                                ),
                                tfidf_cfg,
                            )
                            if text_features is not None
                            else None
                        ),
                        categorical_columns=cats_arg,
                        numerical_columns=nums_arg,
                        model_kwargs=kwargs,
                        model_desc=model_desc,
                        random_seed=random_seed,
                        log_results=log_results,
                    )
                    _add_row(results, model_desc, summary)
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


def print_results_table(results_df, title="MODEL PERFORMANCE COMPARISON", top_n=10):
    """
    Print a results table showing top-performing models.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame of results.
    title : str, default="MODEL PERFORMANCE COMPARISON"
        Table title.
    top_n : int, default=10
        Number of top rows to display.

    Returns
    -------
    None
    """

    print("\n" + "=" * 80)
    print(f"        {title} (Top {top_n})")
    print("=" * 80)
    print(results_df.head(top_n).to_string(index=True))
    print("=" * 80)
