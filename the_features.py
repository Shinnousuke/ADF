import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Time Series Analysis App")

st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    # Load CSV
    df = pd.read_csv(file)

    st.subheader("Raw Data")
    st.write(df.head())

    # Rename first column to YEAR
    df.rename(columns={df.columns[0]: "YEAR"}, inplace=True)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df.dropna(subset=["YEAR"], inplace=True)

    # Use PeriodIndex for years
    df.set_index(pd.PeriodIndex(df["YEAR"], freq="Y"), inplace=True)
    df.drop(columns=["YEAR"], inplace=True)

    # Select column for analysis
    col = st.selectbox("Select column for time series analysis", df.columns)

    # Convert selected series to numeric
    series = pd.to_numeric(df[col], errors="coerce").dropna()

    st.subheader("Selected Series")
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax)
    ax.set_title(f"{col} (1901–2021)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    # -------------------
    # Stationarity Test
    # -------------------
    st.subheader("Stationarity Test (ADF)")
    try:
        result = adfuller(series, autolag='AIC')
        st.write("ADF Statistic:", result[0])
        st.write("p-value:", result[1])
        if result[1] < 0.05:
            st.success("Series is likely stationary")
        else:
            st.warning("Series is not stationary")
    except Exception as e:
        st.error(f"ADF test failed: {e}")

    # -------------------
    # ACF & PACF
    # -------------------
    st.subheader("Autocorrelation and Partial Autocorrelation")
    try:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(series, ax=ax[0])
        plot_pacf(series, ax=ax[1])
        st.pyplot(fig)
    except Exception as e:
        st.error(f"ACF/PACF plotting failed: {e}")

    # -------------------
    # Trend, Seasonality, Cyclic, Irregular Components
    # -------------------
    st.subheader("Decomposition of Time Series")
    try:
        # Seasonal decomposition (for yearly data, freq can be set manually)
        decomposition = seasonal_decompose(series, model="additive", period=10)

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        decomposition.observed.plot(ax=axes[0], title="Original Series", color='blue')
        decomposition.trend.plot(ax=axes[1], title="Trend", color='orange')
        decomposition.seasonal.plot(ax=axes[2], title="Seasonality", color='green')
        decomposition.resid.plot(ax=axes[3], title="Irregular (Residuals)", color='red')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **Explanation:**
        - **Trend:** Long-term movement or direction in the data.
        - **Seasonality:** Regular, repeating pattern (e.g., yearly temperature variation).
        - **Cyclic:** Longer-term fluctuations (captured partly in trend if no fixed period).
        - **Irregular (Residual):** Random noise or unexplained variation.
        """)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")

else:
    st.info("Please upload a CSV file with YEAR and time series data.")
