import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Network IDS Dashboard",
    page_icon="🔒",
    layout="wide"
)

# ===== LOAD DATA & MODELS =====
@st.cache_data
def load_data():
    train = pd.read_csv('data/train_processed.csv')
    test = pd.read_csv('data/test_processed.csv')
    X_test = pd.read_csv('data/X_test.csv') if 'data/X_test.csv' else None
    results = pd.read_csv('models/results.csv')
    return train, test, results

@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/target_encoder.pkl')
    with open('models/feature_names.json', 'r') as f:
        features = json.load(f)
    return model, scaler, encoder, features

train, test, results = load_data()
model, scaler, encoder, features = load_model()

# ===== SIDEBAR =====
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
st.sidebar.title("🔒 Network IDS")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Data Analysis",
    "🏆 Model Results",
    "🔍 Live Detector",
    "ℹ️ About"
])

# ===== PAGE 1: OVERVIEW =====
if page == "🏠 Overview":
    st.title("🔒 Network Intrusion Detection System")
    st.markdown("### AI-Powered Cyber Attack Detection using Machine Learning")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Records", f"{len(train):,}")
    col2.metric("Test Records", f"{len(test):,}")
    col3.metric("Features Used", f"{len(features)}")
    col4.metric("Best Accuracy", f"{results.iloc[0]['Accuracy']}%")
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 What This System Does
    This ML-powered system analyzes network traffic and classifies it as:
    - ✅ **Normal** — Safe, legitimate traffic
    - 🔴 **DoS** — Denial of Service attacks (flooding the network)
    - 🟡 **Probe** — Surveillance & port scanning
    - 🟠 **R2L** — Remote to Local (unauthorized remote access)
    - 🟣 **U2R** — User to Root (privilege escalation)
    
    ### 🛠️ Tech Stack
    `Python` `Scikit-learn` `XGBoost` `Pandas` `Streamlit`
    """)

# ===== PAGE 2: DATA ANALYSIS =====
elif page == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    # Attack distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attack Category Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db']
        cats = train['attack_category'].value_counts()
        cats.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        for i, (idx, val) in enumerate(cats.items()):
            ax.text(i, val + 500, f'{val:,}', ha='center', fontweight='bold')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Normal vs Attack Traffic")
        fig, ax = plt.subplots(figsize=(8, 5))
        is_attack = train['attack_category'].apply(lambda x: 'Attack' if x != 'Normal' else 'Normal')
        counts = is_attack.value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], explode=[0, 0.05], shadow=True,
               textprops={'fontsize': 14})
        ax.set_title('')
        st.pyplot(fig)
    
    # Protocol distribution
    st.subheader("Protocol & Service Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        proto = train['protocol_type'].value_counts()
        ax.pie(proto, labels=proto.index, autopct='%1.1f%%',
               colors=['#3498db', '#e74c3c', '#f39c12'], shadow=True,
               textprops={'fontsize': 13})
        ax.set_title('Protocol Distribution', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        top_services = train['service'].value_counts().head(10)
        top_services.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title('Top 10 Services', fontweight='bold')
        ax.set_xlabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Statistics table
    st.subheader("Statistics per Attack Category")
    stats = train.groupby('attack_category').agg(
        Records=('duration', 'count'),
        Avg_Duration=('duration', 'mean'),
        Avg_Src_Bytes=('src_bytes', 'mean'),
        Avg_Dst_Bytes=('dst_bytes', 'mean')
    ).round(2)
    st.dataframe(stats, use_container_width=True)

# ===== PAGE 3: MODEL RESULTS =====
elif page == "🏆 Model Results":
    st.title("🏆 Model Comparison & Results")
    st.markdown("---")
    
    # Results table
    st.subheader("Performance Comparison")
    st.dataframe(results.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                 color='#2ecc71'), use_container_width=True)
    
    best = results.iloc[0]
    st.success(f"🥇 Best Model: **{best['Model']}** with **{best['Accuracy']}%** Accuracy and **{best['F1-Score']}%** F1-Score")
    
    # Bar chart
    st.subheader("Visual Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = axes[0].bar(range(len(results)), results['Accuracy'], color=colors[:len(results)], edgecolor='black')
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels(results['Model'], rotation=30, ha='right')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0].set_ylim(50, 105)
    for bar, val in zip(bars, results['Accuracy']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val}%', ha='center', fontweight='bold')
    
    x = np.arange(len(results))
    w = 0.2
    for i, m in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
        axes[1].bar(x + i*w, results[m], w, label=m, edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x + w*1.5)
    axes[1].set_xticklabels(results['Model'], rotation=30, ha='right')
    axes[1].set_ylabel('Score (%)')
    axes[1].set_title('All Metrics', fontweight='bold')
    axes[1].legend()
    axes[1].set_ylim(50, 105)
    plt.tight_layout()
    st.pyplot(fig)

# ===== PAGE 4: LIVE DETECTOR =====
elif page == "🔍 Live Detector":
    st.title("🔍 Live Network Traffic Detector")
    st.markdown("Enter network connection features to classify the traffic:")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duration = st.number_input("Duration (seconds)", 0, 100000, 0)
        src_bytes = st.number_input("Source Bytes", 0, 1000000, 491)
        dst_bytes = st.number_input("Destination Bytes", 0, 1000000, 0)
        land = st.selectbox("Land", [0, 1])
        wrong_fragment = st.number_input("Wrong Fragments", 0, 10, 0)
    
    with col2:
        hot = st.number_input("Hot Indicators", 0, 100, 0)
        logged_in = st.selectbox("Logged In", [0, 1])
        num_compromised = st.number_input("Num Compromised", 0, 100, 0)
        root_shell = st.selectbox("Root Shell", [0, 1])
        count = st.number_input("Connection Count (2s)", 0, 600, 2)
    
    with col3:
        srv_count = st.number_input("Service Count (2s)", 0, 600, 2)
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)
        dst_host_count = st.number_input("Dst Host Count", 0, 255, 150)
        dst_host_srv_count = st.number_input("Dst Host Srv Count", 0, 255, 25)
    
    st.markdown("---")
    
    if st.button("🔍 DETECT THREAT", use_container_width=True):
        # Create input with all features, fill missing with 0
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        
        # Map user inputs to features
        user_inputs = {
            'duration': duration, 'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
            'land': land, 'wrong_fragment': wrong_fragment, 'hot': hot,
            'logged_in': logged_in, 'num_compromised': num_compromised,
            'root_shell': root_shell, 'count': count, 'srv_count': srv_count,
            'serror_rate': serror_rate, 'same_srv_rate': same_srv_rate,
            'dst_host_count': dst_host_count, 'dst_host_srv_count': dst_host_srv_count
        }
        
        for key, val in user_inputs.items():
            if key in input_data.columns:
                input_data[key] = val
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        label = encoder.inverse_transform([prediction])[0]
        
        # Display result
        st.markdown("---")
        if label == 'Normal':
            st.success(f"## ✅ SAFE — Normal Traffic")
            st.balloons()
        elif label == 'DoS':
            st.error(f"## 🚨 THREAT DETECTED — DoS Attack!")
            st.warning("Denial of Service: Attacker is flooding the network to make it unavailable.")
        elif label == 'Probe':
            st.warning(f"## ⚠️ THREAT DETECTED — Probe Attack!")
            st.info("Surveillance/Scanning: Attacker is scanning the network for vulnerabilities.")
        elif label == 'R2L':
            st.error(f"## 🚨 THREAT DETECTED — R2L Attack!")
            st.warning("Remote to Local: Attacker is trying to gain unauthorized access remotely.")
        elif label == 'U2R':
            st.error(f"## 🔴 CRITICAL — U2R Attack!")
            st.warning("User to Root: Attacker is trying to escalate privileges to root/admin!")
    
    # Quick test buttons
    st.markdown("---")
    st.subheader("🧪 Quick Test Examples")
    st.markdown("""
    Try these scenarios:
    - **Normal HTTP:** Duration=0, Src Bytes=491, Logged In=1, Same Service Rate=1.0
    - **DoS Attack:** Duration=0, Src Bytes=0, SYN Error Rate=1.0, Count=500
    - **Probe Scan:** Duration=2000, Src Bytes=100000, Count=1, Service Count=1
    """)

# ===== PAGE 5: ABOUT =====
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Network Intrusion Detection System (IDS)
    
    **Author:** *Momin Ahmad*  
    **Program:** BS Data Science — Semester V  
    **University:** *Institute of Management Sciences Peshawar*  
    **Date:** September 2026
    
    ---
    
    ### 📌 Project Overview
    This project builds an intelligent IDS that uses Machine Learning to detect 
    malicious network traffic. It was built as part of my semester project covering:
    - **Computer Networks** — Understanding network protocols & attacks
    - **Machine Learning** — Training classification models
    - **Data Mining** — Discovering patterns in network data
    - **Advanced Statistics** — Statistical validation of results
    
    ### 📦 Dataset
    **NSL-KDD** — A benchmark dataset for network intrusion detection containing 
    125,973 training records with 41 features describing network connections.
    
    ### 🛠️ Tech Stack
    - Python, Pandas, NumPy
    - Scikit-learn, XGBoost
    - Matplotlib, Seaborn
    - Streamlit
    
    ### 📧 Contact
    - LinkedIn: *https://www.linkedin.com/in/momin-ahmad-/*
    - GitHub: *https://github.com/Momin-Khannn*
    """)

# ===== FOOTER =====
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("BS Data Science — Semester V")