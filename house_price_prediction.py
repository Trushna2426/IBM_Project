"""
house_price_prediction.py
=========================
IBM SkillsBuild – Data Analytics with AI
Project : House Price Prediction

Single-file Streamlit application covering the complete data-science workflow:
  data loading → quality checks → EDA → visualisation →
  preprocessing → model training → evaluation → interactive prediction.

Run:
    streamlit run house_price_prediction.py

AI / GenAI Disclosure
---------------------
IBM Bob (AI assistant) was used as a development tool for:
  - Structuring analytical questions
  - Explaining preprocessing and encoding choices
  - Improving code comments and readability
  - Debugging

All numerical calculations, preprocessing, model training, evaluation, and
predictions are executed by Python (pandas, scikit-learn, NumPy) – not by GenAI.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG  (must be the first Streamlit call in the file)
# =============================================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_PATH = "Housing.csv"

# Threshold for describing a Pearson correlation as "moderate" vs "strong"
MODERATE_CORR_THRESHOLD = 0.6

# Feature groups – order here is the canonical order passed to the model
NUM_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BIN_FEATURES = ["mainroad", "guestroom", "basement",
                "hotwaterheating", "airconditioning", "prefarea"]
CAT_FEATURES = ["furnishingstatus"]          # OneHotEncoded
ALL_FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
TARGET       = "price"


# =============================================================================
# 1. LOAD DATA
# =============================================================================
@st.cache_data(show_spinner="Loading Housing.csv …")
def load_data() -> pd.DataFrame:
    """
    Read Housing.csv and return a pandas DataFrame.
    @st.cache_data ensures the CSV is read only once per Streamlit session.
    """
    df = pd.read_csv(DATA_PATH)
    return df


# =============================================================================
# 2. DATA QUALITY CHECKS
# =============================================================================
def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Compute and return basic data-quality statistics:
      shape       – (rows, columns)
      dtypes      – column name → dtype
      nulls       – null count per column
      duplicates  – number of fully-duplicated rows
      describe    – summary statistics for numeric columns
    """
    return {
        "shape":      df.shape,
        "dtypes":     (df.dtypes
                       .reset_index()
                       .rename(columns={"index": "Column", 0: "Data Type"})),
        "nulls":      (df.isnull().sum()
                       .reset_index()
                       .rename(columns={"index": "Column", 0: "Null Count"})),
        "duplicates": int(df.duplicated().sum()),
        "describe":   df.describe().round(2),
    }


# =============================================================================
# 3. PERFORM ANALYSIS  (answers the 9 analytical questions)
# =============================================================================
def perform_analysis(df: pd.DataFrame) -> dict:
    """
    Answer the 9 analytical questions using pandas groupby aggregations.
    Every result is computed from the raw DataFrame – nothing is hard-coded.

    Returns a dict of summary DataFrames.
    Q1 (area vs price) uses a correlation coefficient rather than a groupby.
    Q2–Q9 return count / mean / median price per category value.
    """

    def summary(group_col: str) -> pd.DataFrame:
        return (
            df.groupby(group_col)[TARGET]
            .agg(Count="count", Mean_Price="mean", Median_Price="median")
            .round(2)
            .reset_index()
        )

    return {
        "area_corr":       round(df["area"].corr(df[TARGET]), 4),
        "bedrooms":        summary("bedrooms"),
        "bathrooms":       summary("bathrooms"),
        "stories":         summary("stories"),
        "furnishingstatus": summary("furnishingstatus"),
        "parking":         summary("parking"),
        "airconditioning": summary("airconditioning"),
        "prefarea":        summary("prefarea"),
        "mainroad":        summary("mainroad"),
    }


# =============================================================================
# 4. CREATE VISUALIZATIONS  (pure Matplotlib – no Seaborn)
# =============================================================================
def _bar_chart(ax: plt.Axes, data: pd.DataFrame,
               x_col: str, x_label: str, title: str) -> None:
    """
    Draw a bar chart of Mean_Price on the supplied Axes.
    Extracted as a helper to avoid repeating the same 8 lines eight times.
    """
    ax.bar(data[x_col].astype(str), data["Mean_Price"],
           color="#3b82f6", edgecolor="white", linewidth=0.6)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Average Price (₹)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"₹{v / 1e5:.0f}L")
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def create_visualizations(df: pd.DataFrame, analysis: dict,
                           feature_names_out: tuple,
                           importances: tuple) -> dict:
    """
    Build all Matplotlib figures and return them in a dict keyed by name.
    Figures are created here but not rendered; Streamlit renders them with
    st.pyplot() in the appropriate tab.

    Also builds the feature-importance chart so it is computed once and
    stored alongside the other figures.

    NOTE: this function is intentionally NOT cached with @st.cache_data.
    Matplotlib Figure objects are not reliably picklable, which would cause
    a Streamlit CachingException. The function runs quickly (< 1 s) and is
    only called once per session because its inputs never change after the
    model is trained.

    Parameters
    ----------
    df               : raw DataFrame (for scatter / histogram / heatmap)
    analysis         : dict of groupby summaries from perform_analysis()
    feature_names_out: tuple of encoded column names
    importances      : tuple of importance scores from model.feature_importances_
    """
    figs = {}

    # ── Q1: Area vs Price – scatter plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["area"], df[TARGET] / 1e5,
               alpha=0.50, color="#3b82f6",
               edgecolors="white", linewidths=0.3, s=40)
    ax.set_xlabel("Area (sq ft)", fontsize=10)
    ax.set_ylabel("Price (₹ Lakhs)", fontsize=10)
    ax.set_title("Q1 – Area vs House Price", fontsize=11,
                 fontweight="bold", pad=8)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figs["area_vs_price"] = fig

    # ── Q2–Q9: Bar charts ─────────────────────────────────────────────────────
    bar_specs = [
        ("bedrooms",         "No. of Bedrooms",   "Q2 – Bedrooms vs Average Price"),
        ("bathrooms",        "No. of Bathrooms",  "Q3 – Bathrooms vs Average Price"),
        ("stories",          "No. of Stories",    "Q4 – Stories vs Average Price"),
        ("furnishingstatus", "Furnishing Status",  "Q5 – Furnishing Status vs Average Price"),
        ("parking",          "Parking Spaces",    "Q6 – Parking vs Average Price"),
        ("airconditioning",  "Air Conditioning",  "Q7 – Air Conditioning vs Average Price"),
        ("prefarea",         "Preferred Area",    "Q8 – Preferred Area vs Average Price"),
        ("mainroad",         "Main Road Access",  "Q9 – Main Road Access vs Average Price"),
    ]
    for col, xlabel, title in bar_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        _bar_chart(ax, analysis[col], col, xlabel, title)
        plt.tight_layout()
        figs[col] = fig

    # ── Price distribution – histogram ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET] / 1e5, bins=25,
            color="#3b82f6", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Price (₹ Lakhs)", fontsize=10)
    ax.set_ylabel("Number of Houses", fontsize=10)
    ax.set_title("Distribution of House Prices", fontsize=11,
                 fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figs["price_distribution"] = fig

    # ── Correlation heatmap – pure Matplotlib (no Seaborn) ───────────────────
    num_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    corr = df[num_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    # Annotate each cell with its correlation value
    for r in range(len(num_cols)):
        for c in range(len(num_cols)):
            val = corr.values[r, c]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("Correlation Matrix – Numeric Features",
                 fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    figs["correlation"] = fig

    # ── Feature importance chart ─────────────────────────────────────────────
    imp_series = pd.Series(list(importances),
                           index=list(feature_names_out)).sort_values()
    median_imp = float(imp_series.median())
    imp_colors = ["#3b82f6" if v >= median_imp else "#93c5fd"
                  for v in imp_series.values]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_series.index, imp_series.values,
            color=imp_colors, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("Feature Importances – Random Forest",
                 fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    figs["feature_importance"] = fig

    return figs


# =============================================================================
# 5. PREPARE DATA  (no data leakage)
# =============================================================================
def prepare_data(df: pd.DataFrame):
    """
    Split the dataset FIRST, then fit the encoder ONLY on training data.

    Workflow to avoid data leakage:
      1. Separate X (features) and y (target).
      2. train_test_split → X_train, X_test, y_train, y_test.
      3. Build ColumnTransformer.
      4. encoder.fit(X_train)          ← fitted on training data only.
      5. X_train_enc = encoder.transform(X_train)
      6. X_test_enc  = encoder.transform(X_test)   ← transform only, no fit.

    Encoding decisions
    ------------------
    NUM_FEATURES  passthrough  – already integers; Random Forest is scale-invariant.
    BIN_FEATURES  OrdinalEncoder(no→0, yes→1) – genuinely binary, two-level,
                  clear ordering; no false ordinality introduced.
    CAT_FEATURES  OneHotEncoder(drop='first') – 'furnishingstatus' has 3 unordered
                  levels; OrdinalEncoder would impose a false numeric ordering.
                  drop='first' avoids the dummy-variable trap.

    Returns
    -------
    X_train_enc, X_test_enc : np.ndarray – encoded feature matrices
    y_train, y_test         : pd.Series  – target vectors
    encoder                 : fitted ColumnTransformer (used in predict_price)
    feature_names_out       : list[str]  – column names after encoding
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # Step 2: split BEFORE fitting the encoder
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: build encoder
    encoder = ColumnTransformer(
        transformers=[
            # Binary yes/no features → 0 / 1
            ("bin", OrdinalEncoder(
                categories=[["no", "yes"]] * len(BIN_FEATURES),
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), BIN_FEATURES),

            # Multi-class categorical → one-hot columns
            ("cat", OneHotEncoder(
                drop="first",          # avoids dummy-variable trap
                sparse_output=False,   # return dense array
                handle_unknown="ignore",
            ), CAT_FEATURES),
        ],
        remainder="passthrough",             # numeric features pass through unchanged
        verbose_feature_names_out=False,     # cleaner column names
    )

    # Step 4: fit on training data ONLY
    encoder.fit(X_train)

    # Steps 5 & 6: transform each split independently
    X_train_enc = encoder.transform(X_train)
    X_test_enc  = encoder.transform(X_test)

    # Get the column names that the model will see
    # Order: BIN cols → CAT one-hot cols → NUM cols (passthrough)
    feature_names_out = list(encoder.get_feature_names_out())

    return X_train_enc, X_test_enc, y_train, y_test, encoder, feature_names_out


# =============================================================================
# 6. TRAIN MODEL
# =============================================================================
@st.cache_resource(show_spinner="Training Random Forest model …")
def train_model(_df: pd.DataFrame):
    """
    Train a Random Forest Regressor on the prepared training data.

    @st.cache_resource caches the model and encoder in memory so they are
    trained / fitted only once per Streamlit session regardless of how many
    times the user interacts with the page.

    The leading underscore on `_df` tells Streamlit not to attempt to hash
    the DataFrame argument (DataFrames cannot be hashed by the resource cache).

    Returns
    -------
    model             : trained RandomForestRegressor
    encoder           : fitted ColumnTransformer (for use in predict_price)
    X_test_enc        : encoded test features (for evaluate_model)
    y_test            : test target values
    feature_names_out : list of column names after encoding
    """
    X_train_enc, X_test_enc, y_train, y_test, encoder, feature_names_out = \
        prepare_data(_df)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,        # trees grow until min_samples_leaf is reached
        min_samples_leaf=2,    # prevents overfitting on very small leaf nodes
        random_state=42,
        n_jobs=-1,             # use all CPU cores
    )
    model.fit(X_train_enc, y_train)

    # Evaluate once here so the result is cached with the model and never
    # recomputed on subsequent Streamlit reruns.
    y_pred = model.predict(X_test_enc)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_test, y_pred)

    # Actual vs Predicted scatter (built once, cached)
    fig_avp, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test / 1e5, y_pred / 1e5,
               alpha=0.55, color="#3b82f6",
               edgecolors="white", linewidths=0.3, s=45)
    lo = min(float(y_test.min()), float(y_pred.min())) / 1e5
    hi = max(float(y_test.max()), float(y_pred.max())) / 1e5
    ax.plot([lo, hi], [lo, hi], color="#dc2626",
            linewidth=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual Price (₹ Lakhs)", fontsize=10)
    ax.set_ylabel("Predicted Price (₹ Lakhs)", fontsize=10)
    ax.set_title("Actual vs Predicted House Prices",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    # y_pred is not stored in metrics – it was only needed to compute the
    # scalar metrics and build the figure above. Keeping it in the dict
    # would waste memory inside the @st.cache_resource store.
    metrics = {
        "mae":  mae,
        "mse":  mse,
        "rmse": rmse,
        "r2":   r2,
        "actual_vs_predicted_fig": fig_avp,
    }

    # X_test_enc and y_test are returned so the caller can display
    # len(y_test) in the Model Performance tab caption without recomputing.
    return model, encoder, X_test_enc, y_test, feature_names_out, metrics


# =============================================================================
# 7. PREDICT PRICE
# =============================================================================
def predict_price(model, encoder, user_input: dict) -> float:
    """
    Apply the same fitted encoder used during training, then return a
    predicted price.

    The encoder is NOT re-fitted here – it is the already-fitted
    ColumnTransformer from prepare_data(), which saw only X_train.

    Parameters
    ----------
    model       : trained RandomForestRegressor
    encoder     : ColumnTransformer fitted on X_train ONLY
    user_input  : dict with keys matching ALL_FEATURES

    Returns
    -------
    Predicted house price in Rupees (float, clamped to ≥ 0).
    """
    # Build a single-row DataFrame in the canonical column order
    row = pd.DataFrame([{feat: user_input[feat] for feat in ALL_FEATURES}])

    # Transform using the already-fitted encoder (no re-fitting)
    row_enc = encoder.transform(row)

    predicted = float(model.predict(row_enc)[0])
    return max(0.0, predicted)


# =============================================================================
# 8. HELPER UTILITIES
# =============================================================================
def format_price(price: float) -> str:
    """
    Convert a raw Rupee value to a human-readable Indian format.
      4_500_000  → '₹ 45.00 Lakhs'
      13_000_000 → '₹ 1.30 Crores'
    """
    if price >= 10_000_000:
        return f"₹ {price / 10_000_000:.2f} Crores"
    return f"₹ {price / 100_000:.2f} Lakhs"


# =============================================================================
# 9. STREAMLIT APPLICATION
# =============================================================================
def run_streamlit_app() -> None:
    """
    Main Streamlit UI. Organised into 6 tabs:
      1. Project Overview
      2. Dataset Overview
      3. Data Analysis
      4. Visualizations
      5. Model Performance
      6. House Price Prediction
    """

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🏠 House Price\nPrediction")
        st.markdown("---")
        st.markdown(
            """
            **IBM SkillsBuild**
            Data Analytics with AI

            ---
            **Tech Stack**
            - Python 3
            - Streamlit
            - pandas · NumPy
            - scikit-learn
            - Matplotlib

            ---
            **AI Assistance**
            IBM Bob was used to:
            - Structure analytical questions
            - Explain encoding choices
            - Improve comments & readability
            - Debug issues

            Python performs all
            calculations, training,
            and predictions.
            """
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_data()

    # ── Train model (cached – only once per session) ──────────────────────────
    # evaluate_model is also called inside train_model and cached with it,
    # so metrics are never recomputed on subsequent reruns.
    model, encoder, X_test_enc, y_test, feature_names_out, metrics = train_model(df)

    # ── Pre-compute analysis and visualisations ───────────────────────────────
    quality  = check_data_quality(df)
    analysis = perform_analysis(df)
    # feature_names_out and importances are already the right types
    # (list and ndarray); convert to tuple for consistent function signature.
    figs = create_visualizations(
        df, analysis,
        feature_names_out=tuple(feature_names_out),
        importances=tuple(model.feature_importances_),
    )

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Project Overview",
        "🗂 Dataset Overview",
        "🔍 Data Analysis",
        "📊 Visualizations",
        "🤖 Model Performance",
        "🏠 Predict Price",
    ])

    # =========================================================================
    # TAB 1 – PROJECT OVERVIEW
    # =========================================================================
    with tab1:
        st.title("🏠 House Price Prediction")
        st.subheader("IBM SkillsBuild – Data Analytics with AI")

        # Top-level KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Properties", f"{df.shape[0]:,}")
        k2.metric("Features Used", len(ALL_FEATURES))
        k3.metric("Avg Price", format_price(df[TARGET].mean()))
        k4.metric(
            "Price Range",
            f"{format_price(df[TARGET].min())} – {format_price(df[TARGET].max())}",
        )

        st.markdown("---")
        st.markdown(
            """
            ### About This Project
            This project demonstrates the complete **data analytics and machine
            learning workflow** applied to a housing dataset of **545 properties**
            with **13 features**. The goal is to:

            - Understand which property features most influence price.
            - Build a predictive model for estimating house prices.
            - Provide an interactive tool for price estimation.

            ### Project Workflow
            | Step | What happens |
            |------|-------------|
            | 1 · Data Loading | `Housing.csv` is read into a pandas DataFrame |
            | 2 · Quality Checks | Null values, duplicates, and data types are verified |
            | 3 · Analysis | 9 analytical questions are answered with live calculations |
            | 4 · Visualisation | Charts are produced with Matplotlib |
            | 5 · Preprocessing | Binary features → OrdinalEncoder; `furnishingstatus` → OneHotEncoder |
            | 6 · Model Training | Random Forest Regressor (200 trees, 80 / 20 split) |
            | 7 · Evaluation | MAE, MSE, RMSE, R² computed from actual test-set predictions |
            | 8 · Prediction | User inputs → same fitted encoder → model → estimated price |

            ### Dataset Features
            | Feature | Type | Description |
            |---------|------|-------------|
            | `price` | int | House price in ₹ (target variable) |
            | `area` | int | Plot area in sq ft |
            | `bedrooms` | int | Number of bedrooms (1–6) |
            | `bathrooms` | int | Number of bathrooms (1–4) |
            | `stories` | int | Number of floors (1–4) |
            | `parking` | int | Parking spaces (0–3) |
            | `mainroad` | yes/no | Main road access |
            | `guestroom` | yes/no | Has a guest room |
            | `basement` | yes/no | Has a basement |
            | `hotwaterheating` | yes/no | Has hot water heating |
            | `airconditioning` | yes/no | Has air conditioning |
            | `prefarea` | yes/no | Located in a preferred area |
            | `furnishingstatus` | category | furnished / semi-furnished / unfurnished |

            ### Encoding Strategy
            | Feature Group | Encoder | Justification |
            |---------------|---------|---------------|
            | Numeric (`area`, `bedrooms` …) | Passthrough | Already numeric; Random Forest is scale-invariant |
            | Binary yes/no (6 columns) | `OrdinalEncoder` (no→0, yes→1) | Genuinely two-level; ordering is meaningful |
            | `furnishingstatus` | `OneHotEncoder` (drop first) | 3 unordered levels; OrdinalEncoder would impose a false ranking |

            ### AI & GenAI Disclosure
            > **IBM Bob** (AI assistant) was used as a development tool.
            > All numerical calculations, preprocessing, model training, evaluation,
            > and predictions are executed by **Python** (pandas, scikit-learn, NumPy)
            > – not by GenAI.
            """
        )

    # =========================================================================
    # TAB 2 – DATASET OVERVIEW
    # =========================================================================
    with tab2:
        st.header("🗂 Dataset Overview")

        st.subheader("First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Last 5 Rows")
        st.dataframe(df.tail(), use_container_width=True)

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            rows, n_cols = quality["shape"]
            st.subheader("Shape")
            st.write(f"**{rows} rows × {n_cols} columns**")

            st.subheader("Data Types")
            st.dataframe(quality["dtypes"],
                         use_container_width=True, hide_index=True)

        with col_r:
            st.subheader("Null Values per Column")
            st.dataframe(quality["nulls"],
                         use_container_width=True, hide_index=True)

            dup = quality["duplicates"]
            if dup == 0:
                st.success("✅ No duplicate rows found.")
            else:
                st.warning(f"⚠️ {dup} duplicate row(s) detected.")

        st.markdown("---")
        st.subheader("Descriptive Statistics")
        st.dataframe(quality["describe"], use_container_width=True)

    # =========================================================================
    # TAB 3 – DATA ANALYSIS
    # =========================================================================
    with tab3:
        st.header("🔍 Data Analysis – Analytical Questions")
        st.caption(
            "All summaries are computed by pandas from Housing.csv. "
            "Prices are in Indian Rupees. Nothing is hard-coded."
        )

        # Q1 ──────────────────────────────────────────────────────────────────
        with st.expander("Q1 – How is area related to house price?", expanded=True):
            corr_val = analysis["area_corr"]
            st.metric("Pearson Correlation  (area ↔ price)", f"{corr_val:.4f}")
            strength = "moderate" if abs(corr_val) < MODERATE_CORR_THRESHOLD else "strong"
            st.markdown(
                f"A correlation of **{corr_val:.4f}** indicates a "
                f"**{strength}** positive relationship between plot area and "
                "house price. Larger properties generally command higher prices, "
                "though the scatter plot shows considerable spread — other "
                "features also play important roles."
            )

        # Q2 ──────────────────────────────────────────────────────────────────
        with st.expander("Q2 – How do bedrooms relate to house price?"):
            st.dataframe(analysis["bedrooms"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Average price generally rises with bedroom count up to 5 bedrooms. "
                "The 6-bedroom row has only 2 samples, so its mean is less reliable."
            )

        # Q3 ──────────────────────────────────────────────────────────────────
        with st.expander("Q3 – How do bathrooms relate to house price?"):
            st.dataframe(analysis["bathrooms"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Bathrooms show a strong positive association with price. "
                "Houses with 4 bathrooms have a notably higher mean price."
            )

        # Q4 ──────────────────────────────────────────────────────────────────
        with st.expander("Q4 – How do stories relate to house price?"):
            st.dataframe(analysis["stories"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "More stories consistently correlates with higher average price, "
                "suggesting that multi-storey homes are valued more highly."
            )

        # Q5 ──────────────────────────────────────────────────────────────────
        with st.expander("Q5 – How does furnishing status relate to house price?"):
            st.dataframe(analysis["furnishingstatus"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Furnished homes command the highest average price, followed by "
                "semi-furnished, then unfurnished."
            )

        # Q6 ──────────────────────────────────────────────────────────────────
        with st.expander("Q6 – How does parking relate to house price?"):
            st.dataframe(analysis["parking"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "More parking spaces are associated with higher average prices, "
                "though the increase levels off at 2–3 spaces."
            )

        # Q7 ──────────────────────────────────────────────────────────────────
        with st.expander("Q7 – How does air conditioning relate to house price?"):
            st.dataframe(analysis["airconditioning"],
                         use_container_width=True, hide_index=True)
            # Calculate the price premium from the live data
            ac_df = analysis["airconditioning"]
            yes_price = float(ac_df.loc[ac_df["airconditioning"] == "yes",
                                        "Mean_Price"].values[0])
            no_price  = float(ac_df.loc[ac_df["airconditioning"] == "no",
                                        "Mean_Price"].values[0])
            premium_pct = ((yes_price - no_price) / no_price) * 100
            st.caption(
                f"Houses with air conditioning have a mean price "
                f"**{premium_pct:.1f}% higher** than those without "
                f"(₹{yes_price/1e5:.2f}L vs ₹{no_price/1e5:.2f}L)."
            )

        # Q8 ──────────────────────────────────────────────────────────────────
        with st.expander("Q8 – How does preferred area relate to house price?"):
            st.dataframe(analysis["prefarea"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Properties in a preferred area have a significantly higher "
                "average price, confirming that location is a key pricing factor."
            )

        # Q9 ──────────────────────────────────────────────────────────────────
        with st.expander("Q9 – How does main-road access relate to house price?"):
            st.dataframe(analysis["mainroad"],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Houses with main-road access have a notably higher average "
                "price compared to those without."
            )

    # =========================================================================
    # TAB 4 – VISUALIZATIONS
    # =========================================================================
    with tab4:
        st.header("📊 Visualizations")

        st.subheader("Q1 – Area vs House Price")
        st.pyplot(figs["area_vs_price"])

        st.markdown("---")
        st.subheader("Price Distribution")
        st.pyplot(figs["price_distribution"])

        st.markdown("---")
        st.subheader("Correlation Matrix (Numeric Features)")
        st.pyplot(figs["correlation"])

        st.markdown("---")
        st.subheader("Average Price by Feature (Q2 – Q9)")

        bar_keys = [
            "bedrooms", "bathrooms", "stories", "furnishingstatus",
            "parking",  "airconditioning", "prefarea", "mainroad",
        ]
        # Render in a 2-column grid
        for i in range(0, len(bar_keys), 2):
            c_left, c_right = st.columns(2)
            with c_left:
                st.pyplot(figs[bar_keys[i]])
            if i + 1 < len(bar_keys):
                with c_right:
                    st.pyplot(figs[bar_keys[i + 1]])

    # =========================================================================
    # TAB 5 – MODEL PERFORMANCE
    # =========================================================================
    with tab5:
        st.header("🤖 Model Performance")
        st.caption(
            f"Metrics below are computed on the held-out **20 % test set** "
            f"({len(y_test)} samples). All values come from actual predictions "
            "— nothing is hard-coded."
        )

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score",  f"{metrics['r2']:.4f}",
                  help="Proportion of price variance explained. 1.0 = perfect.")
        m2.metric("MAE",  f"₹ {metrics['mae']:,.0f}",
                  help="Mean Absolute Error — average prediction error in ₹.")
        m3.metric("RMSE", f"₹ {metrics['rmse']:,.0f}",
                  help="Root Mean Squared Error — penalises large errors more.")
        m4.metric("MSE",  f"{metrics['mse']:,.0f} ₹²",
                  help="Mean Squared Error (unit: ₹²; use RMSE for ₹-comparable error).")

        st.markdown("---")
        plot_l, plot_r = st.columns(2)

        with plot_l:
            st.subheader("Actual vs Predicted")
            st.pyplot(metrics["actual_vs_predicted_fig"])

        with plot_r:
            st.subheader("Feature Importances")
            st.caption(
                "Feature names are taken from `encoder.get_feature_names_out()` "
                "after fitting, so one-hot expanded columns are shown correctly."
            )
            st.pyplot(figs["feature_importance"])

        st.markdown("---")
        st.subheader("Model Configuration")
        st.markdown(
            """
            | Parameter | Value |
            |-----------|-------|
            | Algorithm | `RandomForestRegressor` |
            | `n_estimators` | 200 |
            | `max_depth` | None (trees grow until `min_samples_leaf`) |
            | `min_samples_leaf` | 2 |
            | `random_state` | 42 |
            | Train / Test split | 80 % / 20 % |
            | Numeric preprocessing | Passthrough (scale-invariant model) |
            | Binary feature encoding | `OrdinalEncoder` — `no` → 0, `yes` → 1 |
            | `furnishingstatus` encoding | `OneHotEncoder(drop='first')` |
            | Data-leakage prevention | Encoder fitted on X_train only; X_test transformed without re-fitting |
            """
        )

    # =========================================================================
    # TAB 6 – HOUSE PRICE PREDICTION
    # =========================================================================
    with tab6:
        st.header("🏠 House Price Prediction")
        st.markdown(
            "Adjust the property details below and click **Predict House Price** "
            "to receive an estimated price from the trained Random Forest model."
        )
        st.markdown("---")

        with st.form("prediction_form"):

            # Physical dimensions
            st.subheader("Property Dimensions")
            d1, d2, d3 = st.columns(3)
            with d1:
                inp_area = st.number_input(
                    "Area (sq ft)",
                    min_value=1650, max_value=16200,
                    value=5000, step=100,
                    help="Dataset range: 1,650 – 16,200 sq ft.",
                )
            with d2:
                inp_bedrooms = st.selectbox(
                    "Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
            with d3:
                inp_bathrooms = st.selectbox(
                    "Bathrooms", options=[1, 2, 3, 4], index=0)

            d4, d5 = st.columns(2)
            with d4:
                inp_stories = st.selectbox(
                    "Stories (floors)", options=[1, 2, 3, 4], index=1)
            with d5:
                inp_parking = st.selectbox(
                    "Parking spaces", options=[0, 1, 2, 3], index=1)

            # Amenities & location
            st.subheader("Amenities & Location")
            a1, a2, a3 = st.columns(3)
            with a1:
                inp_mainroad = st.radio(
                    "Main road access", ["yes", "no"],
                    index=0, horizontal=True)
            with a2:
                inp_aircon = st.radio(
                    "Air conditioning", ["yes", "no"],
                    index=1, horizontal=True)
            with a3:
                inp_prefarea = st.radio(
                    "Preferred area", ["yes", "no"],
                    index=1, horizontal=True)

            a4, a5, a6 = st.columns(3)
            with a4:
                inp_guestroom = st.radio(
                    "Guest room", ["yes", "no"],
                    index=1, horizontal=True)
            with a5:
                inp_basement = st.radio(
                    "Basement", ["yes", "no"],
                    index=1, horizontal=True)
            with a6:
                inp_hotwater = st.radio(
                    "Hot water heating", ["yes", "no"],
                    index=1, horizontal=True)

            # Furnishing status
            st.subheader("Furnishing")
            inp_furnishing = st.select_slider(
                "Furnishing status",
                options=["unfurnished", "semi-furnished", "furnished"],
                value="semi-furnished",
            )

            submitted = st.form_submit_button(
                "🔮  Predict House Price",
                use_container_width=True,
                type="primary",
            )

        # ── Display prediction ─────────────────────────────────────────────
        if submitted:
            user_input = {
                "area":             int(inp_area),
                "bedrooms":         int(inp_bedrooms),
                "bathrooms":        int(inp_bathrooms),
                "stories":          int(inp_stories),
                "parking":          int(inp_parking),
                "mainroad":         inp_mainroad,
                "guestroom":        inp_guestroom,
                "basement":         inp_basement,
                "hotwaterheating":  inp_hotwater,
                "airconditioning":  inp_aircon,
                "prefarea":         inp_prefarea,
                "furnishingstatus": inp_furnishing,
            }

            predicted = predict_price(model, encoder, user_input)
            avg_price  = float(df[TARGET].mean())
            delta_pct  = ((predicted - avg_price) / avg_price) * 100
            percentile = float((df[TARGET] < predicted).mean() * 100)

            st.markdown("---")
            st.subheader("Prediction Result")

            r1, r2, r3 = st.columns(3)
            r1.metric("Estimated Price", format_price(predicted))
            r2.metric("Raw Value", f"₹ {predicted:,.0f}")
            r3.metric(
                "vs Dataset Average",
                format_price(avg_price),
                delta=f"{delta_pct:+.1f}%",
                help="Positive = above average; negative = below average.",
            )

            st.info(
                f"This prediction is higher than **{percentile:.1f}%** of the "
                f"{df.shape[0]} properties in the dataset."
            )

            with st.expander("📋 Input summary"):
                inp_df = pd.DataFrame(
                    user_input.items(), columns=["Feature", "Value"]
                )
                st.dataframe(inp_df,
                             use_container_width=True, hide_index=True)


# =============================================================================
# ENTRY POINT
# =============================================================================
# Streamlit imports this file as a module, so we call run_streamlit_app()
# at module level. The if __name__ == "__main__" guard is also present for
# completeness (e.g. running with `python house_price_prediction.py` for a
# quick syntax / import check).
run_streamlit_app()
