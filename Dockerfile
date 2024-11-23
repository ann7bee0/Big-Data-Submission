FROM bitnami/spark:latest

# Install Python, NumPy, Pandas, and Streamlit
USER root
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install numpy && \
    pip3 install pandas && \
    pip3 install streamlit && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER 1001