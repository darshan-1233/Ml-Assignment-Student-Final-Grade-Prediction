import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide"
)


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    rf_model = joblib.load(
        "random_forest_model.pkl"
    )

    svr_model = joblib.load(
        "svr_model.pkl"
    )

    comparison = joblib.load(
        "model_comparison.pkl"
    )

    return rf_model, svr_model, comparison


rf_model, svr_model, comparison = load_models()


# =====================================================
# TITLE
# =====================================================

st.title("🎓 Student Final Grade Prediction")

st.markdown(
    """
    ### Machine Learning Application

    This application predicts a student's final grade (**G3**)
    using two Machine Learning algorithms:

    - 🌲 Random Forest Regressor
    - 📈 Support Vector Regression (SVR)

    The same student information is provided to both models.
    """
)


st.divider()


# =====================================================
# STUDENT INPUT FORM
# =====================================================

st.header("📝 Enter Student Information")


col1, col2, col3 = st.columns(3)


with col1:

    school = st.selectbox(
        "School",
        ["GP", "MS"]
    )

    sex = st.selectbox(
        "Gender",
        ["F", "M"]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=25,
        value=17
    )

    address = st.selectbox(
        "Address",
        ["U", "R"]
    )

    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"]
    )

    Pstatus = st.selectbox(
        "Parent Status",
        ["A", "T"]
    )

    Medu = st.slider(
        "Mother Education",
        0,
        4,
        2
    )

    Fedu = st.slider(
        "Father Education",
        0,
        4,
        2
    )


with col2:

    Mjob = st.selectbox(
        "Mother Job",
        [
            "teacher",
            "health",
            "services",
            "at_home",
            "other"
        ]
    )

    Fjob = st.selectbox(
        "Father Job",
        [
            "teacher",
            "health",
            "services",
            "at_home",
            "other"
        ]
    )

    reason = st.selectbox(
        "Reason for Choosing School",
        [
            "home",
            "reputation",
            "course",
            "other"
        ]
    )

    guardian = st.selectbox(
        "Guardian",
        [
            "mother",
            "father",
            "other"
        ]
    )

    traveltime = st.slider(
        "Travel Time",
        1,
        4,
        1
    )

    studytime = st.slider(
        "Study Time",
        1,
        4,
        2
    )

    failures = st.slider(
        "Previous Failures",
        0,
        4,
        0
    )

    schoolsup = st.selectbox(
        "Extra School Support",
        ["yes", "no"]
    )


with col3:

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extra Activities",
        ["yes", "no"]
    )

    nursery = st.selectbox(
        "Attended Nursery School",
        ["yes", "no"]
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"]
    )

    famrel = st.slider(
        "Family Relationship Quality",
        1,
        5,
        4
    )


# =====================================================
# MORE INPUTS
# =====================================================

st.subheader("Lifestyle and Academic Information")

col4, col5, col6 = st.columns(3)


with col4:

    freetime = st.slider(
        "Free Time",
        1,
        5,
        3
    )

    goout = st.slider(
        "Going Out",
        1,
        5,
        3
    )

    Dalc = st.slider(
        "Workday Alcohol Consumption",
        1,
        5,
        1
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1,
        5,
        1
    )


with col5:

    health = st.slider(
        "Health",
        1,
        5,
        3
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=4
    )

    G1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    G2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )


st.divider()


# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🔮 Predict Final Grade",
    use_container_width=True
):

    # ---------------------------------------------
    # ENGINEERED FEATURES
    # ---------------------------------------------

    academic_average = (
        G1 + G2
    ) / 2

    alcohol_index = (
        Dalc + Walc
    )

    social_index = (
        freetime + goout
    )

    support_index = (
        int(schoolsup == "yes") +
        int(famsup == "yes")
    )


    # ---------------------------------------------
    # CREATE INPUT DATAFRAME
    # ---------------------------------------------

    input_data = pd.DataFrame({

        "school": [school],
        "sex": [sex],
        "age": [age],
        "address": [address],
        "famsize": [famsize],
        "Pstatus": [Pstatus],

        "Medu": [Medu],
        "Fedu": [Fedu],

        "Mjob": [Mjob],
        "Fjob": [Fjob],

        "reason": [reason],
        "guardian": [guardian],

        "traveltime": [traveltime],
        "studytime": [studytime],
        "failures": [failures],

        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "nursery": [nursery],
        "higher": [higher],
        "internet": [internet],
        "romantic": [romantic],

        "famrel": [famrel],
        "freetime": [freetime],
        "goout": [goout],

        "Dalc": [Dalc],
        "Walc": [Walc],

        "health": [health],
        "absences": [absences],

        "G1": [G1],
        "G2": [G2],

        "academic_average": [
            academic_average
        ],

        "alcohol_index": [
            alcohol_index
        ],

        "social_index": [
            social_index
        ],

        "support_index": [
            support_index
        ]
    })


    # ---------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------

    rf_prediction = rf_model.predict(
        input_data
    )[0]

    svr_prediction = svr_model.predict(
        input_data
    )[0]


    # Keep grade between 0 and 20

    rf_prediction = np.clip(
        rf_prediction,
        0,
        20
    )

    svr_prediction = np.clip(
        svr_prediction,
        0,
        20
    )


    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------

    st.header("📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.subheader(
            "🌲 Random Forest"
        )

        st.metric(
            "Predicted Final Grade",
            f"{rf_prediction:.2f} / 20"
        )


    with result_col2:

        st.subheader(
            "📈 Support Vector Regression"
        )

        st.metric(
            "Predicted Final Grade",
            f"{svr_prediction:.2f} / 20"
        )


    # ---------------------------------------------
    # GRADE INTERPRETATION
    # ---------------------------------------------

    average_prediction = (
        rf_prediction +
        svr_prediction
    ) / 2


    if average_prediction >= 16:

        interpretation = "Excellent"

    elif average_prediction >= 12:

        interpretation = "Good"

    elif average_prediction >= 10:

        interpretation = "Average"

    else:

        interpretation = "Needs Improvement"


    st.info(
        f"Overall predicted performance: **{interpretation}**"
    )


    # ---------------------------------------------
    # PREDICTION COMPARISON CHART
    # ---------------------------------------------

    st.subheader(
        "📈 Prediction Comparison"
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    models = [
        "Random Forest",
        "SVR"
    ]

    predictions = [
        rf_prediction,
        svr_prediction
    ]

    ax.bar(
        models,
        predictions
    )

    ax.set_ylabel(
        "Predicted G3"
    )

    ax.set_ylim(
        0,
        20
    )

    ax.set_title(
        "Predicted Final Grade by Algorithm"
    )

    st.pyplot(fig)


    # ---------------------------------------------
    # MODEL PERFORMANCE
    # ---------------------------------------------

    st.subheader(
        "🏆 Model Performance Comparison"
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )


    # ---------------------------------------------
    # FIND BETTER MODEL
    # ---------------------------------------------

    best_model = comparison.loc[
        comparison["R2"].idxmax(),
        "Model"
    ]

    st.success(
        f"🏆 Better Performing Algorithm: **{best_model}**"
    )


    # ---------------------------------------------
    # PERFORMANCE CHART
    # ---------------------------------------------

    st.subheader(
        "📊 R² Model Comparison"
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    ax2.bar(
        comparison["Model"],
        comparison["R2"]
    )

    ax2.set_ylabel(
        "R² Score"
    )

    ax2.set_title(
        "Model Performance Based on R²"
    )

    st.pyplot(fig2)