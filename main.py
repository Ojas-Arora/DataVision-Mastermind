import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import requests
from streamlit_lottie import st_lottie

# Load Lottie Animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

penguin_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# Set page config
st.set_page_config(page_title="ML DATASET COMPARISON", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-image: url('https://wallpapers.com/images/hd/abstract-blueish-white-color-nrvpjoky2673bptv.jpg');
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
    h1, h2, h3, h4, h5, h6{
        color: black;
        text-align: center;
        text-transform: uppercase;       
    }
    label{
        text-align: center;       
    }
    p{
        font-size:20px;
        text-align: center;          
    }
    .sidebar-content {
        color: white !important;
    }
    .sidebar-content > div {
        color: white !important;
    }
    .stAlert {
        font-size: 1.2rem;
        text-align:center;
        padding: 1rem;
        border-radius: 0.25rem;
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
    .sidebar-content {
        color: white ;
    }
    .sidebar-content > div {
        color: white ;
    }
    </style>
""", unsafe_allow_html=True)

st.title('📊 ML DATASET COMPARISON')

# Lottie animation for penguins
st_lottie(penguin_animation, speed=1, height=400, key="penguin")

st.write("""
🌟 Discover the ultimate machine learning model for your dataset! Dive into our interactive tool to compare top classifiers and see which one reigns supreme. 

🔍 Compare top classifiers across diverse datasets to find the best performer for your needs.
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

# Plotting Accuracy vs Parameter for SVM
if classifier_name == 'SVM':
    param_range = np.linspace(0.01, 10.0, 10)
    accuracy_values = []
    for param in param_range:
        clf = SVC(C=param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_values.append(acc)
    
    fig_param, ax_param = plt.subplots()
    ax_param.plot(param_range, accuracy_values, marker='o')
    ax_param.set_xlabel('C')
    ax_param.set_ylabel('Accuracy')
    ax_param.set_title('Accuracy vs C for SVM')
    st.pyplot(fig_param)

# Confusion Matrix
fig_cm, ax_cm = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax_cm.figure.colorbar(im, ax=ax_cm)
ax_cm.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          ylabel='True label',
          xlabel='Predicted label',
          title='Confusion Matrix')
plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
plt.tight_layout()

st.pyplot(fig_cm)

# Adding more visual elements
st.markdown("### 🎯KEY FEATURES")
st.write("""
- **INTERACTIVE WIDGETS:** Select datasets and classifiers from the sidebar to dynamically update the content.
- **PERFORMANCE METRICS:** View accuracy scores and confusion matrices to evaluate model performance.
- **VISUALIZATION:** PCA visualization of dataset classes for easy interpretation.
""")

st.markdown("### 🧠 HOW TO USE")
st.write("""
1. **SELECT A DATASET:** Choose from Iris, Breast Cancer, or Wine datasets.
2. **CHOOSE A CLASSIFIER:** Options include K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest.
3. **SET PARAMETERS:** Adjust the hyperparameters for each classifier using the sliders in the sidebar.
4. **VIEW RESULTS:** See the accuracy score, PCA scatter plot, and confusion matrix of the dataset.
""")

# Adding icons
st.sidebar.markdown("## OPTIONS")
st.sidebar.write("🎯 SELECT DATASET")
st.sidebar.write("🧠 SELECT CLASSIFIER")

st.sidebar.write("## HYPERPARAMETERS")
st.sidebar.write("⚙️ ADJUST PARAMETERS")

st.markdown("---")
st.write("Developed with ❤️ using Streamlit")
st.markdown("---")
st.markdown("---")
# Adding footer
st.markdown("""
    <style>
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
    </style>
    <div class="footer">
        <p>Developed by Ojas Arora with ❤️ using <a href="https://streamlit.io/" target="_blank">Streamlit</a></p>
    </div>
""", unsafe_allow_html=True)
