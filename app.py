import os
import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Detective",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# PROFESSIONAL DARK / BLUE UI THEME
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 85% 5%, rgba(76, 82, 190, 0.18), transparent 28%),
                    linear-gradient(135deg, #070b16 0%, #0b1020 48%, #10152b 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080d1d 0%, #0c1225 100%);
        border-right: 1px solid rgba(117, 128, 255, 0.20);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .side-brand {
        padding: 0 0 1.1rem 0;
    }

    .side-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .side-subtitle {
        margin-top: 0.35rem;
        color: #9ba7c7;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .side-list {
        display: grid;
        gap: 0.45rem;
        color: #cdd5f2;
        font-size: 0.83rem;
        line-height: 1.35;
    }

    .tech-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }

    .tech-list span {
        border: 1px solid rgba(130, 145, 255, 0.24);
        background: rgba(87, 100, 190, 0.10);
        border-radius: 999px;
        padding: 0.25rem 0.5rem;
        color: #bfc9ee;
        font-size: 0.72rem;
    }

    .hero {
        padding: 2.25rem 2.4rem;
        margin: 0.5rem 0 1.5rem 0;
        border-radius: 24px;
        border: 1px solid rgba(130, 145, 255, 0.24);
        background: linear-gradient(120deg, rgba(33, 83, 196, 0.95) 0%, rgba(83, 61, 190, 0.96) 52%, rgba(126, 55, 177, 0.94) 100%);
        box-shadow: 0 22px 60px rgba(22, 35, 110, 0.28);
    }

    .hero-kicker {
        color: #d9e5ff;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.14em;
    }

    .hero-title {
        margin-top: 0.4rem;
        color: white;
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -0.05em;
    }

    .hero-text {
        max-width: 760px;
        margin-top: 0.8rem;
        color: #e9edff;
        font-size: 1rem;
        line-height: 1.6;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.25rem;
    }

    .hero-badges span {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        color: white;
        font-size: 0.78rem;
        font-weight: 700;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(113, 143, 255, 0.55);
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(37, 45, 83, 0.62), rgba(20, 25, 49, 0.72));
        padding: 0.65rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(27, 34, 64, 0.92), rgba(15, 20, 40, 0.92));
        border: 1px solid rgba(113, 127, 210, 0.18);
        border-radius: 16px;
        padding: 0.9rem 1rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(113, 143, 255, 0.34);
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    hr {
        border-color: rgba(113, 127, 210, 0.16);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_value(value):
    if pd.isna(value):
        return "No value"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"

    return str(value)


def get_iqr_anomalies(dataframe):
    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    anomaly_columns = []
    anomaly_counts = {}
    total_anomalies = 0

    for column in numeric_columns:

        values = dataframe[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        if count > 0:
            anomaly_columns.append(column)
            anomaly_counts[column] = count
            total_anomalies += count

    return (
        anomaly_columns,
        anomaly_counts,
        total_anomalies
    )


def get_strongest_correlation(dataframe):

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) < 2:
        return None

    correlation = dataframe[numeric_columns].corr()

    pairs = []

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            column_a = numeric_columns[i]
            column_b = numeric_columns[j]

            value = correlation.loc[
                column_a,
                column_b
            ]

            if not pd.isna(value):

                pairs.append(
                    (
                        abs(value),
                        column_a,
                        column_b,
                        value
                    )
                )

    if not pairs:
        return None

    pairs.sort(reverse=True)

    return pairs[0][1], pairs[0][2], pairs[0][3]


# ============================================================
# PROFESSIONAL UI
# ============================================================

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="side-brand">
        <div class="side-title">🕵️ AI Data Detective</div>
        <div class="side-subtitle">AI-powered data quality & anomaly investigation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Features")
    st.markdown("""
    <div class="side-list">
        <div>📊 Dataset Overview</div>
        <div>🚨 Missing Value Detection</div>
        <div>🔁 Duplicate Detection</div>
        <div>🕵️ IQR Anomaly Detection</div>
        <div>🤖 ML Anomaly Detection</div>
        <div>🧠 Smart Data Insights</div>
        <div>📈 Interactive Visualization</div>
        <div>🧹 Data Cleaning</div>
        <div>❤️ Dataset Health Score</div>
        <div>📄 PDF Data Reports</div>
        <div>💬 Local AI Chat with Qwen</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology")
    st.markdown("""
    <div class="tech-list">
        <span>Streamlit</span>
        <span>Pandas / NumPy</span>
        <span>Scikit-learn</span>
        <span>Plotly</span>
        <span>Ollama + Qwen 2.5 3B</span>
        <span>ReportLab</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🎯 Turn Data into Actionable Insights")
    st.caption("Built as a portfolio / placement project")

# Main hero
st.markdown("""
<div class="hero">
    <div class="hero-kicker">🔎 AI-POWERED DATA ANALYTICS</div>
    <div class="hero-title">AI Data Detective</div>
    <div class="hero-text">Upload your dataset and automatically discover data quality issues, anomalies, patterns, and actionable insights.</div>
    <div class="hero-badges">
        <span>📊 Analyze</span>
        <span>🚨 Detect</span>
        <span>🧠 Explain</span>
        <span>📄 Report</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is None:

    st.info(
        "👆 Upload a CSV or Excel file to start."
    )

    st.markdown(
        """
### 🔎 AI Data Detective Features

- 📊 Dataset Overview
- 🚨 Missing Value Detection
- 🔁 Duplicate Detection
- 🕵️ IQR Anomaly Detection
- 🤖 Machine Learning Anomaly Detection
- 🧠 Smart Data Insights
- 📈 Data Visualization
- 🧹 Data Cleaning
- ❤️ Dataset Health Score
- 📊 Data Quality Dashboard
- 🔎 Interactive Data Explorer
- 📄 Automatic Data Report
- 🤖 Real AI Chatbox
        """
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

except Exception as error:

    st.error(
        f"❌ Could not read the dataset: {error}"
    )

    st.stop()


st.success(
    "✅ Dataset loaded successfully!"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

total_rows = len(df)

total_columns = len(df.columns)

total_missing = int(
    df.isna().sum().sum()
)

total_duplicates = int(
    df.duplicated().sum()
)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        total_rows
    )

with col2:

    st.metric(
        "Columns",
        total_columns
    )

with col3:

    st.metric(
        "Missing Values",
        total_missing
    )

with col4:

    st.metric(
        "Duplicate Rows",
        total_duplicates
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("👀 Data Preview")

st.dataframe(
    df,
    width="stretch"
)


# ============================================================
# COLUMN INFORMATION
# ============================================================

st.subheader("📋 Column Information")

column_info = pd.DataFrame({

    "Column":
        df.columns.astype(str),

    "Data Type":
        df.dtypes.astype(str).values,

    "Missing Values":
        df.isna().sum().values,

    "Unique Values":
        df.nunique().values
})

st.dataframe(
    column_info,
    width="stretch"
)


# ============================================================
# MISSING VALUES
# ============================================================

st.subheader("🚨 Missing Values")

missing_values = df.isna().sum()

missing_values = missing_values[
    missing_values > 0
]

if not missing_values.empty:

    st.warning(
        f"⚠️ {total_missing} missing value(s) found."
    )

    st.dataframe(
        missing_values.rename(
            "Missing Count"
        ).to_frame(),
        width="stretch"
    )

else:

    st.success(
        "✅ No missing values found."
    )


# ============================================================
# DUPLICATES
# ============================================================

st.subheader("🔁 Duplicate Detection")

if total_duplicates > 0:

    st.warning(
        f"⚠️ {total_duplicates} duplicate row(s) found."
    )

else:

    st.success(
        "✅ No duplicate rows found."
    )


# ============================================================
# IQR ANOMALY DETECTION
# ============================================================

st.subheader("🕵️ IQR Anomaly Detection")

(
    anomaly_columns,
    anomaly_counts,
    iqr_anomaly_total
) = get_iqr_anomalies(df)


if anomaly_columns:

    for column in anomaly_columns:

        st.warning(
            f"⚠️ {column}: "
            f"{anomaly_counts[column]} "
            f"unusual value(s) detected."
        )

        values = df[column].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        unusual = df[
            (df[column] < lower)
            |
            (df[column] > upper)
        ]

        st.dataframe(
            unusual[[column]],
            width="stretch"
        )

else:

    st.success(
        "✅ No obvious numeric anomalies found."
    )


# ============================================================
# DATASET HEALTH SCORE
# ============================================================

st.subheader("❤️ Dataset Health Score")

total_cells = (
    total_rows * total_columns
)

missing_penalty = (

    total_missing
    / total_cells
    * 40

    if total_cells > 0
    else 0
)


duplicate_penalty = (

    total_duplicates
    / total_rows
    * 20

    if total_rows > 0
    else 0
)


anomaly_penalty = (
    len(anomaly_columns) * 10
)


health_score = round(
    max(
        0,
        100
        - missing_penalty
        - duplicate_penalty
        - anomaly_penalty
    )
)


st.progress(
    health_score / 100
)


if health_score >= 80:

    st.success(
        f"🟢 Dataset Health Score: "
        f"{health_score}/100"
    )

elif health_score >= 60:

    st.warning(
        f"🟡 Dataset Health Score: "
        f"{health_score}/100"
    )

else:

    st.error(
        f"🔴 Dataset Health Score: "
        f"{health_score}/100"
    )


# ============================================================
# SMART DATA INSIGHTS
# ============================================================

st.subheader("🧠 Smart Data Insights")

insights = []


# Missing data insight

if not missing_values.empty:

    highest_missing_column = (
        missing_values.idxmax()
    )

    highest_missing_count = int(
        missing_values.max()
    )

    insights.append(
        f"⚠️ **Missing data:** "
        f"`{highest_missing_column}` has the "
        f"highest number of missing values "
        f"({highest_missing_count})."
    )

else:

    insights.append(
        "✅ **Missing data:** "
        "No missing values were found."
    )


# Duplicate insight

if total_duplicates > 0:

    insights.append(
        f"🔁 **Duplicates:** "
        f"The dataset contains "
        f"**{total_duplicates} duplicate row(s)**."
    )

else:

    insights.append(
        "✅ **Duplicates:** "
        "No duplicate rows were found."
    )


# Numeric statistics

for column in numeric_columns:

    minimum = df[column].min()

    maximum = df[column].max()

    average = df[column].mean()

    insights.append(
        f"📈 **{column}:** "
        f"minimum = **{format_value(minimum)}**, "
        f"maximum = **{format_value(maximum)}**, "
        f"average = **{format_value(average)}**."
    )


# Strongest correlation

correlation_result = (
    get_strongest_correlation(df)
)


if correlation_result:

    column_a, column_b, correlation_value = (
        correlation_result
    )

    insights.append(
        f"🔗 **Strongest numeric relationship:** "
        f"`{column_a}` and `{column_b}` have "
        f"a correlation of "
        f"**{correlation_value:.2f}**."
    )


# Anomaly insight

if anomaly_columns:

    insights.append(
        f"🔎 **Potential anomalies:** "
        f"unusual values were detected in "
        f"`{', '.join(anomaly_columns)}` "
        f"using the IQR method."
    )

else:

    insights.append(
        "✅ **Potential anomalies:** "
        "No IQR-based unusual values were detected."
    )


for insight in insights:

    st.write(insight)


plain_insights = "\n\n".join(

    re.sub(
        r"\*\*",
        "",
        insight
    )

    for insight in insights
)


st.download_button(

    "📥 Download Smart Insights",

    data=plain_insights,

    file_name="smart_data_insights.txt",

    mime="text/plain"
)


# ============================================================
# VISUAL ANALYSIS
# ============================================================

st.subheader("📈 Visual Analysis")


if numeric_columns:

    selected_numeric = st.selectbox(

        "Choose a numeric column",

        numeric_columns,

        key="histogram_column"
    )


    histogram = px.histogram(

        df,

        x=selected_numeric,

        title=(
            f"Distribution of "
            f"{selected_numeric}"
        ),

        nbins=20
    )


    st.plotly_chart(

        histogram,

        width="stretch"
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

if len(numeric_columns) >= 2:

    st.subheader(
        "🔗 Correlation Analysis"
    )

    correlation = (
        df[numeric_columns].corr()
    )


    heatmap = px.imshow(

        correlation,

        text_auto=True,

        title="Numeric Feature Correlation"
    )


    st.plotly_chart(

        heatmap,

        width="stretch"
    )


# ============================================================
# CATEGORICAL CHART
# ============================================================

if categorical_columns:

    st.subheader(
        "📊 Categorical Analysis"
    )


    selected_category = st.selectbox(

        "Choose a categorical column",

        categorical_columns,

        key="category_column"
    )


    category_counts = (

        df[selected_category]

        .fillna("Missing")

        .value_counts()

        .head(20)

        .reset_index()
    )


    category_counts.columns = [

        selected_category,

        "Count"
    ]


    bar_chart = px.bar(

        category_counts,

        x=selected_category,

        y="Count",

        title=(
            f"Distribution of "
            f"{selected_category}"
        )
    )


    st.plotly_chart(

        bar_chart,

        width="stretch"
    )


# ============================================================
# SCATTER PLOT
# ============================================================

if len(numeric_columns) >= 2:

    st.subheader(
        "🔵 Numeric Relationship"
    )


    x_column = st.selectbox(

        "X-axis",

        numeric_columns,

        key="scatter_x"
    )


    y_options = [

        column

        for column in numeric_columns

        if column != x_column
    ]


    if y_options:

        y_column = st.selectbox(

            "Y-axis",

            y_options,

            key="scatter_y"
        )


        scatter = px.scatter(

            df,

            x=x_column,

            y=y_column,

            title=(
                f"{x_column} vs {y_column}"
            )
        )


        st.plotly_chart(

            scatter,

            width="stretch"
        )


# ============================================================
# DATA CLEANING
# ============================================================

st.subheader(
    "🧹 Data Cleaning"
)


cleaned_df = df.copy()


remove_duplicates = st.checkbox(

    "Remove duplicate rows",

    value=True
)


fill_missing = st.checkbox(

    "Fill missing numeric values "
    "with column median",

    value=True
)


if st.button(
    "🧹 Clean Dataset"
):

    if remove_duplicates:

        cleaned_df = (
            cleaned_df
            .drop_duplicates()
        )


    if fill_missing:

        clean_numeric_columns = (
            cleaned_df
            .select_dtypes(
                include=np.number
            )
            .columns
        )


        for column in clean_numeric_columns:

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna(
                    cleaned_df[column].median()
                )
            )


    st.success(
        "✅ Dataset cleaned successfully!"
    )


    st.dataframe(

        cleaned_df,

        width="stretch"
    )


    st.download_button(

        "📥 Download Cleaned CSV",

        data=cleaned_df.to_csv(
            index=False
        ),

        file_name="cleaned_dataset.csv",

        mime="text/csv"
    )


# ============================================================
# MACHINE LEARNING ANOMALY DETECTION
# ============================================================

st.subheader(
    "🤖 ML Anomaly Detection"
)


ml_anomaly_count = 0

ml_anomaly_rows = pd.DataFrame()


if (
    len(numeric_columns) >= 2
    and len(df) >= 5
):

    ml_data = df[
        numeric_columns
    ].copy()


    imputer = SimpleImputer(
        strategy="median"
    )


    ml_data = imputer.fit_transform(
        ml_data
    )


    model = IsolationForest(

        n_estimators=100,

        contamination="auto",

        random_state=42
    )


    predictions = model.fit_predict(
        ml_data
    )


    ml_anomaly_count = int(

        (predictions == -1).sum()
    )


    if ml_anomaly_count > 0:

        st.warning(

            f"🤖 Machine Learning detected "
            f"{ml_anomaly_count} unusual "
            f"record(s)."
        )


        ml_anomaly_rows = df[
            predictions == -1
        ].copy()


        st.dataframe(

            ml_anomaly_rows,

            width="stretch"
        )


    else:

        st.success(

            "🤖 Machine Learning found "
            "no unusual records."
        )

else:

    st.info(

        "At least 2 numeric columns and "
        "5 rows are required for ML "
        "anomaly detection."
    )


# ============================================================
# DATA QUALITY DASHBOARD
# ============================================================

st.subheader(
    "📊 Data Quality Dashboard"
)


d1, d2, d3, d4 = st.columns(4)


with d1:

    st.metric(
        "❌ Missing Values",
        total_missing
    )


with d2:

    st.metric(
        "🔁 Duplicate Rows",
        total_duplicates
    )


with d3:

    st.metric(
        "🔎 IQR Anomalies",
        iqr_anomaly_total
    )


with d4:

    st.metric(
        "❤️ Health Score",
        f"{health_score}/100"
    )


if health_score >= 80:

    st.success(
        "🟢 Dataset quality is good."
    )

elif health_score >= 60:

    st.warning(
        "🟡 Dataset quality needs "
        "some improvement."
    )

else:

    st.error(
        "🔴 Dataset quality needs attention."
    )


# ============================================================
# INTERACTIVE DATA EXPLORER
# ============================================================

st.subheader(
    "🔎 Interactive Data Explorer"
)


search_text = st.text_input(
    "🔍 Search the dataset"
)


selected_columns = st.multiselect(

    "📋 Select columns",

    options=list(df.columns),

    default=list(df.columns)
)


explorer_df = df.copy()


if search_text:

    search_mask = (
        explorer_df
        .astype(str)
        .apply(

            lambda row:
            row.str.contains(
                search_text,
                case=False,
                na=False
            ).any(),

            axis=1
        )
    )


    explorer_df = (
        explorer_df[
            search_mask
        ]
    )


if selected_columns:

    explorer_df = (
        explorer_df[
            selected_columns
        ]
    )

else:

    st.warning(
        "Please select at least one column."
    )


st.dataframe(

    explorer_df,

    width="stretch"
)


st.download_button(

    "📥 Download Filtered CSV",

    data=explorer_df.to_csv(
        index=False
    ),

    file_name="filtered_dataset.csv",

    mime="text/csv"
)


# ============================================================
# AUTOMATIC DATA REPORT
# ============================================================

st.subheader(
    "📄 Automatic Data Report"
)


if numeric_columns:

    numeric_summary = (

        df[numeric_columns]

        .describe()

        .round(2)

        .to_string()
    )

else:

    numeric_summary = (
        "No numeric columns found."
    )


report_text = f"""

AI DATA DETECTIVE
AUTOMATIC DATA REPORT
=====================

DATASET OVERVIEW
----------------

Rows: {total_rows}

Columns: {total_columns}

Column Names:
{", ".join(str(c) for c in df.columns)}


DATA QUALITY
------------

Missing Values: {total_missing}

Duplicate Rows: {total_duplicates}


IQR ANOMALIES
-------------

Total Anomalies: {iqr_anomaly_total}

Affected Columns:
{", ".join(anomaly_columns) if anomaly_columns else "None"}


MACHINE LEARNING
----------------

Unusual Records Detected:
{ml_anomaly_count}


NUMERIC SUMMARY
---------------

{numeric_summary}


DATASET HEALTH
--------------

{health_score}/100


SMART INSIGHTS
--------------

{plain_insights}


RECOMMENDATION
--------------

{
"Dataset quality is good. Continue monitoring data quality."
if health_score >= 80
else
"Review missing values, duplicate rows, and unusual records before analysis."
}

"""


with st.expander(
    "📋 View Full Report"
):

    st.text(
        report_text
    )


st.download_button(

    "⬇️ Download Data Report",

    data=report_text,

    file_name="ai_data_detective_report.txt",

    mime="text/plain"
)


# ============================================================
# ============================================================
# EXECUTIVE DATA DASHBOARD
# ============================================================

st.markdown("---")
st.subheader("📊 Executive Data Dashboard")
st.caption("A quick overview of your dataset health and quality.")

# Basic metrics
total_rows = len(df)
total_columns = len(df.columns)

total_missing = int(df.isnull().sum().sum())
total_duplicates = int(df.duplicated().sum())

# IQR anomaly count
iqr_anomaly_count = 0

numeric_dashboard_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

for col in numeric_dashboard_cols:
    series = df[col].dropna()

    if len(series) >= 4:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            iqr_anomaly_count += int(
                ((series < lower) | (series > upper)).sum()
            )

# Health score
health_score = 100

if total_rows > 0:
    missing_ratio = total_missing / (total_rows * total_columns)

    health_score -= min(30, int(missing_ratio * 100))

if total_duplicates > 0:
    health_score -= min(
        20,
        int((total_duplicates / max(total_rows, 1)) * 100)
    )

if iqr_anomaly_count > 0:
    health_score -= min(
        20,
        int((iqr_anomaly_count / max(total_rows, 1)) * 100)
    )

health_score = max(0, min(100, health_score))

# KPI cards
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("📦 Total Rows", f"{total_rows:,}")

with c2:
    st.metric("📋 Total Columns", f"{total_columns:,}")

with c3:
    st.metric("⚠️ Missing Values", f"{total_missing:,}")

c4, c5, c6 = st.columns(3)

with c4:
    st.metric("🔁 Duplicate Rows", f"{total_duplicates:,}")

with c5:
    st.metric("🚨 IQR Anomalies", f"{iqr_anomaly_count:,}")

with c6:
    st.metric("❤️ Health Score", f"{health_score}/100")

# Health interpretation
if health_score >= 90:
    st.success("🟢 Excellent dataset health")

elif health_score >= 75:
    st.info("🟡 Good dataset health with some issues to review")

elif health_score >= 50:
    st.warning("🟠 Dataset needs cleaning")

else:
    st.error("🔴 Dataset requires significant cleaning")

# Quick summary
st.markdown("### 🔎 Quick Dataset Summary")

summary_items = []

if total_missing > 0:
    summary_items.append(
        f"• {total_missing} missing value(s) detected."
    )
else:
    summary_items.append(
        "• No missing values detected."
    )

if total_duplicates > 0:
    summary_items.append(
        f"• {total_duplicates} duplicate row(s) detected."
    )
else:
    summary_items.append(
        "• No duplicate rows detected."
    )

if iqr_anomaly_count > 0:
    summary_items.append(
        f"• {iqr_anomaly_count} potential IQR anomaly value(s) detected."
    )
else:
    summary_items.append(
        "• No IQR anomalies detected."
    )

for item in summary_items:
    st.write(item)

# ============================================================
# ============================================================
# SMART AI RECOMMENDATIONS
# ============================================================

st.markdown("---")
st.subheader("🧠 Smart Data Recommendations")
st.caption("Automatically generated findings from your dataset.")

smart_findings = []
smart_actions = []

# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

if total_missing > 0:

    missing_columns = (
        df.isnull()
        .sum()
    )

    missing_columns = missing_columns[
        missing_columns > 0
    ]

    for col, count in missing_columns.items():

        percentage = (
            count / len(df)
        ) * 100

        smart_findings.append(
            f"⚠️ Column '{col}' has {int(count)} missing value(s) "
            f"({percentage:.1f}% of the dataset)."
        )

    smart_actions.append(
        "Review missing values and decide whether to fill, remove, "
        "or investigate them."
    )

else:

    smart_findings.append(
        "✅ No missing values were detected."
    )


# ------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------

if total_duplicates > 0:

    smart_findings.append(
        f"🔁 {total_duplicates} duplicate row(s) were detected."
    )

    smart_actions.append(
        "Check duplicate records before using the dataset for "
        "machine learning or reporting."
    )

else:

    smart_findings.append(
        "✅ No duplicate rows were detected."
    )


# ------------------------------------------------------------
# IQR ANOMALIES
# ------------------------------------------------------------

if iqr_anomaly_count > 0:

    smart_findings.append(
        f"🚨 {iqr_anomaly_count} potential IQR anomaly value(s) "
        "were detected."
    )

    smart_actions.append(
        "Investigate unusual numeric values before making "
        "important conclusions."
    )

else:

    smart_findings.append(
        "✅ No IQR anomalies were detected."
    )


# ------------------------------------------------------------
# NUMERIC ANALYSIS
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

if numeric_cols:

    for col in numeric_cols:

        series = df[col].dropna()

        if len(series) > 0:

            average = series.mean()
            minimum = series.min()
            maximum = series.max()

            smart_findings.append(
                f"📊 '{col}': average = {average:.2f}, "
                f"minimum = {minimum:.2f}, "
                f"maximum = {maximum:.2f}."
            )

    # --------------------------------------------------------
    # STRONG CORRELATIONS
    # --------------------------------------------------------

    if len(numeric_cols) >= 2:

        correlation_matrix = df[
            numeric_cols
        ].corr()

        strongest_pair = None
        strongest_value = 0

        for i in range(len(numeric_cols)):

            for j in range(i + 1, len(numeric_cols)):

                value = correlation_matrix.iloc[i, j]

                if pd.notna(value):

                    if abs(value) > abs(strongest_value):

                        strongest_value = value

                        strongest_pair = (
                            numeric_cols[i],
                            numeric_cols[j]
                        )

        if strongest_pair:

            smart_findings.append(
                f"🔗 Strongest numeric relationship: "
                f"'{strongest_pair[0]}' and "
                f"'{strongest_pair[1]}' "
                f"(correlation = {strongest_value:.2f})."
            )

            if abs(strongest_value) >= 0.7:

                smart_actions.append(
                    f"Investigate the relationship between "
                    f"'{strongest_pair[0]}' and "
                    f"'{strongest_pair[1]}'."
                )


# ------------------------------------------------------------
# HEALTH RECOMMENDATION
# ------------------------------------------------------------

if health_score >= 90:

    smart_findings.append(
        "🟢 Overall dataset quality is excellent."
    )

elif health_score >= 75:

    smart_findings.append(
        "🟡 Overall dataset quality is good, "
        "but some issues should be reviewed."
    )

elif health_score >= 50:

    smart_findings.append(
        "🟠 The dataset needs cleaning before deeper analysis."
    )

else:

    smart_findings.append(
        "🔴 The dataset requires significant cleaning."
    )


# ------------------------------------------------------------
# DISPLAY FINDINGS
# ------------------------------------------------------------

st.markdown("### 🔎 Key Findings")

for finding in smart_findings:

    st.write(finding)


st.markdown("### 🎯 Recommended Actions")

if smart_actions:

    for action in smart_actions:

        st.write(f"• {action}")

else:

    st.success(
        "No major corrective actions are currently required."
    )


# ------------------------------------------------------------
# DOWNLOAD SMART INSIGHTS
# ------------------------------------------------------------

smart_report = "AI DATA DETECTIVE - SMART INSIGHTS\n"
smart_report += "=" * 50 + "\n\n"

smart_report += "KEY FINDINGS\n"
smart_report += "-" * 50 + "\n"

for finding in smart_findings:

    smart_report += finding + "\n"

smart_report += "\nRECOMMENDED ACTIONS\n"
smart_report += "-" * 50 + "\n"

for action in smart_actions:

    smart_report += "• " + action + "\n"

smart_report += "\nDATASET HEALTH\n"
smart_report += "-" * 50 + "\n"

smart_report += (
    f"Health Score: {health_score}/100\n"
)

smart_report += (
    f"Rows: {total_rows}\n"
)

smart_report += (
    f"Columns: {total_columns}\n"
)

smart_report += (
    f"Missing Values: {total_missing}\n"
)

smart_report += (
    f"Duplicates: {total_duplicates}\n"
)

smart_report += (
    f"IQR Anomalies: {iqr_anomaly_count}\n"
)


st.download_button(
    "📥 Download Smart Insights",
    data=smart_report,
    file_name="AI_Data_Detective_Smart_Insights.txt",
    mime="text/plain",
    width="content"
)

# PROFESSIONAL PDF DATA REPORT
# ============================================================

from io import BytesIO

st.markdown("---")
st.subheader("📄 Professional Data Report")
st.caption("Generate a downloadable PDF report for this dataset.")

def create_pdf_report():

    buffer = BytesIO()

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak
    )

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=14,
        spaceAfter=8
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI DATA DETECTIVE",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Professional Dataset Analysis Report",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Dataset Overview",
            heading_style
        )
    )

    overview_data = [
        ["Metric", "Value"],
        ["Total Rows", str(len(df))],
        ["Total Columns", str(len(df.columns))],
        ["Missing Values", str(total_missing)],
        ["Duplicate Rows", str(total_duplicates)],
        ["IQR Anomalies", str(iqr_anomaly_count)],
        ["Health Score", f"{health_score}/100"]
    ]

    overview_table = Table(
        overview_data,
        colWidths=[3.2 * inch, 2.5 * inch]
    )

    overview_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303030")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(overview_table)

    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Column Information",
            heading_style
        )
    )

    column_data = [
        ["Column", "Data Type", "Missing", "Unique"]
    ]

    for col in df.columns:

        column_data.append([
            str(col),
            str(df[col].dtype),
            str(int(df[col].isna().sum())),
            str(int(df[col].nunique(dropna=True)))
        ])

    column_table = Table(
        column_data,
        repeatRows=1,
        colWidths=[
            2.0 * inch,
            1.5 * inch,
            1.0 * inch,
            1.0 * inch
        ]
    )

    column_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303030")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 5)
        ])
    )

    story.append(column_table)

    # --------------------------------------------------------
    # NUMERIC STATISTICS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Statistical Summary",
            heading_style
        )
    )

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_cols:

        stat_data = [
            ["Column", "Mean", "Median", "Minimum", "Maximum"]
        ]

        for col in numeric_cols:

            series = df[col].dropna()

            if len(series) > 0:

                stat_data.append([
                    str(col),
                    f"{series.mean():.2f}",
                    f"{series.median():.2f}",
                    f"{series.min():.2f}",
                    f"{series.max():.2f}"
                ])

        stat_table = Table(
            stat_data,
            repeatRows=1,
            colWidths=[
                1.5 * inch,
                1.1 * inch,
                1.1 * inch,
                1.1 * inch,
                1.1 * inch
            ]
        )

        stat_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303030")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 5)
            ])
        )

        story.append(stat_table)

    else:

        story.append(
            Paragraph(
                "No numeric columns were detected.",
                styles["Normal"]
            )
        )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Data Quality Findings",
            heading_style
        )
    )

    if total_missing > 0:
        story.append(
            Paragraph(
                f"• {total_missing} missing value(s) detected.",
                styles["Normal"]
            )
        )
    else:
        story.append(
            Paragraph(
                "• No missing values detected.",
                styles["Normal"]
            )
        )

    if total_duplicates > 0:
        story.append(
            Paragraph(
                f"• {total_duplicates} duplicate row(s) detected.",
                styles["Normal"]
            )
        )
    else:
        story.append(
            Paragraph(
                "• No duplicate rows detected.",
                styles["Normal"]
            )
        )

    if iqr_anomaly_count > 0:
        story.append(
            Paragraph(
                f"• {iqr_anomaly_count} potential IQR anomaly value(s) detected.",
                styles["Normal"]
            )
        )
    else:
        story.append(
            Paragraph(
                "• No IQR anomalies detected.",
                styles["Normal"]
            )
        )

    # --------------------------------------------------------
    # HEALTH SCORE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. Dataset Health",
            heading_style
        )
    )

    if health_score >= 90:
        health_message = "Excellent dataset health."
    elif health_score >= 75:
        health_message = "Good dataset health with some issues to review."
    elif health_score >= 50:
        health_message = "Dataset needs cleaning."
    else:
        health_message = "Dataset requires significant cleaning."

    story.append(
        Paragraph(
            f"Health Score: <b>{health_score}/100</b>",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            health_message,
            styles["Normal"]
        )
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. Recommendations",
            heading_style
        )
    )

    recommendations = []

    if total_missing > 0:
        recommendations.append(
            "Review and handle missing values."
        )

    if total_duplicates > 0:
        recommendations.append(
            "Remove or investigate duplicate records."
        )

    if iqr_anomaly_count > 0:
        recommendations.append(
            "Investigate potential outlier values."
        )

    if not recommendations:
        recommendations.append(
            "Dataset quality looks good. Continue with deeper analysis."
        )

    for recommendation in recommendations:

        story.append(
            Paragraph(
                f"• {recommendation}",
                styles["Normal"]
            )
        )

    # --------------------------------------------------------
    # FOOTER / FINAL NOTE
    # --------------------------------------------------------

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by AI Data Detective",
            subtitle_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# Generate PDF
pdf_data = create_pdf_report()

st.download_button(
    "📥 Download Professional PDF Report",
    data=pdf_data,
    file_name="AI_Data_Detective_Report.pdf",
    mime="application/pdf",
    width="content"
)

# REAL AI CHATBOX

st.subheader("🤖 AI Data Detective Chat")
st.write("Ask the local AI questions about your uploaded dataset.")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"

# ------------------------------------------------------------
# CHAT MEMORY
# ------------------------------------------------------------

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

col_chat, col_clear = st.columns([5, 1])

with col_clear:
    if st.button("🧹 Clear Chat"):
        st.session_state.ai_chat_history = []
        st.rerun()

# ------------------------------------------------------------
# SHOW PREVIOUS MESSAGES
# ------------------------------------------------------------

for message in st.session_state.ai_chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ------------------------------------------------------------
# NEW QUESTION
# ------------------------------------------------------------

question = st.chat_input("Ask AI about your dataset...")

if question:

    import requests

    # Show user question immediately
    with st.chat_message("user"):
        st.write(question)

    st.session_state.ai_chat_history.append({
        "role": "user",
        "content": question
    })


    # ----------------------------------------------------
    # DATASET CONTEXT
    # ----------------------------------------------------

    if len(df) <= 300:
        data_for_ai = df.to_string(index=False)
    else:
        data_for_ai = df.head(100).to_string(index=False)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_cols:
        stats_text = df[numeric_cols].describe(
            include="all"
        ).round(2).to_string()
    else:
        stats_text = "No numeric columns."

    missing_values = df.isnull().sum()
    missing_text = missing_values[
        missing_values > 0
    ].to_string()

    if not missing_text:
        missing_text = "No missing values."

    dataset_context = f"""
You are the AI Data Detective assistant.

You analyze the user's uploaded dataset.

IMPORTANT RULES:
1. Use the supplied dataset information.
2. Never invent dataset values.
3. If the answer cannot be determined, say so.
4. For numerical questions, use the provided statistics.
5. Keep answers clear and concise.
6. When useful, explain how you reached the answer.

DATASET INFORMATION

Rows: {len(df)}
Columns: {len(df.columns)}

Column names:
{", ".join(df.columns.astype(str))}

Missing values:
{missing_text}

Numeric statistics:
{stats_text}

Dataset:
{data_for_ai}
"""

    # ----------------------------------------------------
    # RECENT CONVERSATION
    # ----------------------------------------------------

    recent_history = st.session_state.ai_chat_history[-10:]

    messages = [
        {
            "role": "system",
            "content": dataset_context
        }
    ]

    for message in recent_history:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    # ----------------------------------------------------
    # LOCAL AI
    # ----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 AI is analyzing your dataset..."):

            try:

                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                answer = result["message"]["content"]

                st.write(answer)

            except requests.exceptions.ConnectionError:

                # ------------------------------------------------
                # CLOUD-SAFE FALLBACK
                # ------------------------------------------------
                # Ollama is available locally but not on Streamlit
                # Cloud. Use direct dataframe analysis instead.
        
                question_lower = question.lower()

                numeric_columns = df.select_dtypes(
                    include=np.number
                ).columns.tolist()

                if "row" in question_lower:
                    answer = f"📊 Your dataset contains **{len(df)} rows**."

                elif "column" in question_lower:
                    answer = (
                        f"📊 Your dataset contains **{len(df.columns)} columns**.\n\n"
                        f"Columns: {', '.join(map(str, df.columns))}"
                    )

                elif "missing" in question_lower:
                    missing_count = int(df.isna().sum().sum())

                    if missing_count == 0:
                        answer = "✅ There are **no missing values** in the dataset."
                    else:
                        missing_details = df.isna().sum()
                        missing_details = missing_details[
                            missing_details > 0
                        ]

                        details = ", ".join(
                            f"{col}: {int(count)}"
                            for col, count in missing_details.items()
                        )

                        answer = (
                            f"⚠️ The dataset contains **{missing_count} "
                            f"missing values**.\n\n"
                            f"Details: {details}"
                        )

                elif "duplicate" in question_lower:
                    duplicate_count = int(df.duplicated().sum())

                    if duplicate_count == 0:
                        answer = "✅ No duplicate rows were detected."
                    else:
                        answer = (
                            f"⚠️ The dataset contains "
                            f"**{duplicate_count} duplicate row(s)**."
                        )

                elif (
                    "average" in question_lower
                    or "mean" in question_lower
                ):

                    selected_column = None

                    for column in numeric_columns:
                        if str(column).lower() in question_lower:
                            selected_column = column
                            break

                    if selected_column is not None:

                        value = df[selected_column].mean()

                        answer = (
                            f"📈 The average **{selected_column}** is "
                            f"**{value:.2f}**."
                        )

                    elif numeric_columns:

                        values = df[numeric_columns].mean()

                        answer = "📈 Average values:\n\n"

                        for column, value in values.items():
                            answer += f"- **{column}:** {value:.2f}\n"

                    else:
                        answer = "There are no numeric columns to calculate an average."

                elif (
                    "highest" in question_lower
                    or "maximum" in question_lower
                    or "max" in question_lower
                ):

                    selected_column = None

                    for column in numeric_columns:
                        if str(column).lower() in question_lower:
                            selected_column = column
                            break

                    if selected_column is None and numeric_columns:
                        selected_column = numeric_columns[0]

                    if selected_column is not None:

                        value = df[selected_column].max()

                        answer = (
                            f"🔝 The highest **{selected_column}** is "
                            f"**{value}**."
                        )

                    else:
                        answer = "There are no numeric columns to analyze."

                elif (
                    "lowest" in question_lower
                    or "minimum" in question_lower
                    or "min" in question_lower
                ):

                    selected_column = None

                    for column in numeric_columns:
                        if str(column).lower() in question_lower:
                            selected_column = column
                            break

                    if selected_column is None and numeric_columns:
                        selected_column = numeric_columns[0]

                    if selected_column is not None:

                        value = df[selected_column].min()

                        answer = (
                            f"🔽 The lowest **{selected_column}** is "
                            f"**{value}**."
                        )

                    else:
                        answer = "There are no numeric columns to analyze."

                elif "anomal" in question_lower:

                    anomaly_columns = []

                    for column in numeric_columns:

                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)

                        iqr = q3 - q1

                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr

                        count = int(
                            (
                                (df[column] < lower)
                                | (df[column] > upper)
                            ).sum()
                        )

                        if count > 0:
                            anomaly_columns.append(
                                f"**{column}:** {count} unusual value(s)"
                            )

                    if anomaly_columns:

                        answer = (
                            "🚨 Potential anomalies were detected:\n\n"
                            + "\n".join(
                                f"- {item}"
                                for item in anomaly_columns
                            )
                        )

                    else:
                        answer = "✅ No obvious IQR anomalies were detected."

                elif "health" in question_lower:

                    missing_count = int(df.isna().sum().sum())
                    duplicate_count = int(df.duplicated().sum())

                    anomaly_count = 0

                    for column in numeric_columns:

                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)
                        iqr = q3 - q1

                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr

                        anomaly_count += int(
                            (
                                (df[column] < lower)
                                | (df[column] > upper)
                            ).sum()
                        )

                    health_score = 100

                    if len(df) > 0:
                        health_score -= int(
                            (missing_count / (len(df) * len(df.columns))) * 40
                        )

                        health_score -= int(
                            (duplicate_count / len(df)) * 20
                        )

                        health_score -= min(
                            30,
                            anomaly_count * 10
                        )

                    health_score = max(0, min(100, health_score))

                    answer = (
                        f"💚 Dataset Health Score: **{health_score}/100**\n\n"
                        f"- Missing values: **{missing_count}**\n"
                        f"- Duplicate rows: **{duplicate_count}**\n"
                        f"- Potential anomalies: **{anomaly_count}**"
                    )

                else:

                    answer = (
                        "☁️ The local Qwen AI is not available on the "
                        "deployed cloud version.\n\n"
                        "However, I can still analyze your dataset directly. "
                        "Try questions such as:\n\n"
                        "- How many rows?\n"
                        "- What is the average marks?\n"
                        "- Who has the highest marks?\n"
                        "- What is the lowest age?\n"
                        "- Are there any missing values?\n"
                        "- Are there any anomalies?\n"
                        "- What is the dataset health?"
                    )

                st.write(answer)

            except requests.exceptions.Timeout:

                answer = (
                    "⏱️ The local AI took too long to respond. "
                    "Please try again."
                )

                st.write(answer)

            except Exception as e:

                answer = (
                    "⚠️ AI analysis could not be completed.\n\n"
                    "Please try a dataset question such as "
                    "'What is the average marks?'"
                )

                st.write(answer)

    # Save AI response only after a question was processed
    if question and "answer" in locals():
        st.session_state.ai_chat_history.append({
            "role": "assistant",
            "content": answer
        })