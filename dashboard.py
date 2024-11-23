import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

# Set up environment for Spark
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Commodity Trade Dashboard") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# HDFS Path for the dataset
cleaned_data_path = "hdfs://namenode:8020/cleaned_data/cleaned_data.csv"

# Load the dataset using Spark
try:
    cleaned_data = spark.read.csv(cleaned_data_path, header=True, inferSchema=True)
    df = cleaned_data.limit(10000).toPandas()  # Convert to Pandas DataFrame for Streamlit
except Exception as e:
    st.error(f"Failed to load the dataset from HDFS. Error: {e}")
    spark.stop()
    st.stop()

# Streamlit App Configuration
st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["Prediction", "Visualizations"])

### TAB 1: Prediction ###
with tab1:
    st.title("Commodity Trade Value Prediction")
    st.write("Enter the weight (in tonnes) to predict the trade value (in USD).")

    # Input for weight
    weight_ton = st.number_input("Enter Weight (in Tonnes):", min_value=0.0, step=0.1, format="%.2f")

    if st.button("Predict Trade Value"):
        # Perform simple math-based prediction
        trade_value = weight_ton * 91123  # USD per tonne
        st.success(f"Predicted Trade Value: ${trade_value:,.2f}")

### TAB 2: Visualizations ###
with tab2:
    st.title("Commodity Trade Data Visualizations")
    st.write("Explore insights from the cleaned dataset with descriptive visualizations.")

    # Total Trade Value by Country and Import vs Export Distribution (Two graphs in one row)
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Total Trade Value by Country")
        total_trade_by_country = df.groupby("country")["trade_usd"].sum().sort_values(ascending=False)
        top_countries = total_trade_by_country.head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        top_countries.plot(kind="bar", color="#6495ED", ax=ax)
        ax.set_title("Top 10 Countries by Trade Value", fontsize=10)
        ax.set_ylabel("Trade Value (USD)", fontsize=8)
        ax.set_xlabel("Country", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        st.pyplot(fig)

    with col2:
        st.write("#### Import vs Export Distribution")
        trade_flow_distribution = df["flow"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(6, 3))
        trade_flow_distribution.plot(
            kind="pie",
            startangle=90,
            colors=['#FFA07A', '#20B2AA', '#FFD700', '#FF4500', '#90EE90'],  # Unique colors
            ax=ax,
            textprops={'fontsize': 0},  # Remove labels by setting font size to 0
        )
        ax.legend(trade_flow_distribution.index, loc="upper right", fontsize=4)  # Keep legend
        ax.set_ylabel("")  # Remove default ylabel
        ax.set_title("Import vs Export", fontsize=10)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.write("#### Average Weight (in Tonnes) by Country")
        avg_weight_by_country = df.groupby("country")["weight_ton"].mean().sort_values(ascending=False).head(10) / 1000
        fig, ax = plt.subplots(figsize=(6, 3))
        avg_weight_by_country.plot(kind="bar", color="#8A2BE2", ax=ax)
        ax.set_title("Average Weight by Country", fontsize=10)
        ax.set_ylabel("Weight (Tonnes)", fontsize=8)
        ax.set_xlabel("Country", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        st.pyplot(fig)

    with col4:
        st.write("#### Total Weight Contribution by Commodity")
        weight_by_commodity = df.groupby("commodity")["weight_ton"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        weight_by_commodity.plot(kind="bar", color="#4682B4", ax=ax)
        ax.set_title("Weight by Commodity", fontsize=10)
        ax.set_ylabel("Weight (kg)", fontsize=8)
        ax.set_xlabel("Commodity", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        st.pyplot(fig)

    # Correlation Between Trade Value and Weight
    st.write("#### Correlation Between Trade Value and Weight")
    correlation_matrix = df[["trade_usd", "weight_ton"]].corr()  # Fixed correlation columns
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax, cbar_kws={"shrink": 0.8}, fmt=".2f")
    ax.set_title("Correlation Heatmap (Trade Value vs Weight)", fontsize=10)
    st.pyplot(fig)

    # Yearly Trade Value and Weight Trends
    st.write("#### Yearly Trends of Trade Value and Weight")
    yearly_trends = df.groupby("year")[["trade_usd", "weight_ton"]].sum()
    st.line_chart(yearly_trends)
    st.write("**Description:** Yearly trends of trade value (in USD) and weight (in tonnes) showing the progression of global trade over time.")

    # Top Commodities by Trade Value
    st.write("#### Top Commodities by Trade Value")
    top_commodities = df.groupby("commodity")["trade_usd"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    top_commodities.plot(kind="bar", color="#FF6347", ax=ax)
    ax.set_title("Top 10 Commodities by Trade Value", fontsize=10)
    ax.set_ylabel("Trade Value (USD)", fontsize=8)
    ax.set_xlabel("Commodity", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    st.pyplot(fig)
    st.write("**Description:** Bar chart shows the trade value for the top 10 commodities globally.")
