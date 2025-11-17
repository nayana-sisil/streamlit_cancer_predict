import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_clean_data():
    data = pd.read_csv('data.csv')
    cols_to_drop = ['Unnamed: 32', 'id']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.header("🔬 Cell Nuclei Details")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"), ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"), ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"), ("Texture (se)", "texture_se"), ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"), ("Smoothness (se)", "smoothness_se"), ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"), ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"), ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"), ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"), ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    with st.sidebar.expander("Mean Values", expanded=True):
        for label, key in slider_labels[:10]:
            input_dict[key] = st.slider(label, float(data[key].min()), float(data[key].max()), float(data[key].mean()))

    with st.sidebar.expander("Standard Error Values"):
        for label, key in slider_labels[10:20]:
            input_dict[key] = st.slider(label, float(data[key].min()), float(data[key].max()), float(data[key].mean()))

    with st.sidebar.expander("Worst Values"):
        for label, key in slider_labels[20:]:
            input_dict[key] = st.slider(label, float(data[key].min()), float(data[key].max()), float(data[key].mean()))

    return input_dict

def get_scaled_values_dict(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values_dict(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dim']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean Value',
        line_color='dodgerblue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
           input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
           input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error',
        line_color='mediumseagreen'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
           input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst Value',
        line_color='tomato'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#151a24',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                linecolor='rgba(255, 255, 255, 0.2)',
                gridcolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='white')
            )
        ),
        showlegend=True,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color='white',
            size=14
        ),
        margin=dict(l=50, r=50, b=50, t=50)
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("🩺 Diagnosis Prediction")
    st.write("The predicted cluster analysis:")

    if prediction[0] == 0:
        st.markdown('<div class="diagnosis benign">Benign</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="diagnosis malicious">Malignant</div>', unsafe_allow_html=True)

    st.write("---")

    probs = model.predict_proba(input_array_scaled)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability: Benign", f"{probs[0]:.2%}")
        st.progress(probs[0])

    with col2:
        st.metric("Probability: Malignant", f"{probs[1]:.2%}")
        st.progress(probs[1])

    st.markdown("---")
    st.caption("Disclaimer: This app can assist medical professionals in making a diagnosis, "
               "but should not be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👩‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        with open("assests/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        try:
            with open("assets/style.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Could not find style.css. Please check if it is in 'assests' or 'assets'.")

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Diagnosis Assistant")
        st.write("Connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. "
                 "This app predicts using a machine learning model whether a breast mass is benign or malignant based "
                 "on the measurements it receives from your cytosis lab. You can also update the measurements by hand "
                 "using the sliders in the sidebar.")

    col1, col2 = st.columns([3, 2])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()
