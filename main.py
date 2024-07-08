import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="ML DATASET COMPARISON", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
        background-color: #f0f2f6;
        color: #333;
        font-family: 'Roboto', sans-serif;
    }
    .block-container {
        padding: 2rem;
    }
    .stButton>button {
        background-color: darkturquoise;
        color: white;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: darkturquoise;
    }
    .stTextInput>div>div>input {
        font-size: 1rem;
        text-align:center;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ccc;
    }
    .stSelectbox>div>div>select {
        font-size: 1rem;
        text-align:center;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ccc;
    }
    .stSlider>div>div>div>div>div>div {
        font-size: 1rem;
        text-align:center;
    }
    .stNumberInput>div>div>input {
        font-size: 1rem;
        text-align:center;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ccc;
    }
    .stAlert {
        font-size: 1.2rem;
        text-align:center;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: purple;
        text-align: center;
        text-transform: uppercase;       
    }
    label {
        text-align: center;       
    }
    p {
        font-size: 20px;
        text-align: center;          
    }
    .sidebar-content {
        color: white;
    }
    .sidebar-content > div {
        color: white;
    }
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    .gap {
        margin-bottom: 2rem;
    }
    </style>
    <div class="footer">
        <p>Developed by Ojas Arora with ❤️ using <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
    </div>
""", unsafe_allow_html=True)

st.title('📊 ML DATASET COMPARISON')

st.write("""
Discover the ultimate machine learning model for your dataset! Dive into our interactive tool to compare top classifiers and see which one reigns supreme.

Compare top classifiers across diverse datasets to find the best performer for your needs.
""")

dataset_name = st.sidebar.selectbox(
    '🎯 SELECT DATASET', ('IRIS', 'BREAST CANCER', 'WINE'))

st.write(f"## {dataset_name} DATASET")

classifier_name = st.sidebar.selectbox(
    '🧠 SELECT CLASSIFIER', ('KNN', 'SVM', 'RANDOM FOREST'))

def get_dataset(name):
    data = None
    if name == 'IRIS':
        data = datasets.load_iris()
    elif name == 'WINE':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('**SHAPE OF DATASET:**', X.shape)
st.write('**NUMBER OF CLASSES:**', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'**CLASSIFIER:** {classifier_name}')
st.write(f'**🎯ACCURACY:** {acc}')

# PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig, ax = plt.subplots()
scatter = ax.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter)
st.pyplot(fig)
st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

def plot_histograms(X, y):
    df = pd.DataFrame(X)
    df['target'] = y
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df, x=df.columns[0], hue='target', multiple="stack", ax=ax)
    st.pyplot(fig)
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

def plot_boxplot(X, y):
    df = pd.DataFrame(X)
    df['target'] = y
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='target', y=df.columns[0], data=df, ax=ax)
    st.pyplot(fig)
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

def plot_pairplot(X, y):
    df = pd.DataFrame(X)
    df['target'] = y
    sns.pairplot(df, hue='target', palette='viridis')
    st.pyplot(plt)
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

def plot_correlation_matrix(X):
    df = pd.DataFrame(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

def plot_scatter_matrix(X, y):
    df = pd.DataFrame(X)
    df['target'] = y
    sns.pairplot(df, hue='target', palette='viridis')
    st.pyplot(plt)
    st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

# Call additional graph functions
plot_histograms(X, y)
plot_boxplot(X, y)
plot_pairplot(X, y)
plot_correlation_matrix(X)
plot_scatter_matrix(X, y)

st.markdown("### 🎯KEY FEATURES")
st.write("""
- **INTERACTIVE WIDGETS:** Select datasets and classifiers from the sidebar to dynamically update the content.
- **PERFORMANCE METRICES:** View accuracy scores to evaluate model performance.
- **VISUALIZATION:** PCA visualization of dataset classes for easy interpretation.
""")

st.markdown("### 🧠 HOW TO USE")
st.write("""
1. **SELECT A DATASET:** Choose from Iris, Breast Cancer, or Wine datasets.
2. **CHOOSE A CLASSIFIER:** Options include K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest.
3. **SET PARAMETERS:** Adjust the hyperparameters for each classifier using the sliders in the sidebar.
4. **VIEW RESULTS:** See the accuracy score and a PCA scatter plot of the dataset.
""")
