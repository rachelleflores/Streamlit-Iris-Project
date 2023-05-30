import pandas as pd
import numpy as np
import pickle
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import streamlit as st


iris = datasets.load_iris()

# X-y split
X = iris.data #iris['data']
y = iris.target #iris['target']
cols = iris['feature_names']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1) # default 75-25 split

# modeling
lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_ = SVC()

lin_reg_fit = lin_reg.fit(X_train, y_train)
log_reg_fit = log_reg.fit(X_train, y_train)
svc_fit = svc_.fit(X_train, y_train)


# creating pickle files (saving the models)
with open("lin_reg.pkl", "wb") as li:  # wb: mode write
    pickle.dump(lin_reg_fit, li)

with open("log_reg.pkl", "wb") as lo:
    pickle.dump(log_reg_fit, lo)

with open("svc_.pkl", "wb") as sv:
    pickle.dump(svc_fit, sv)

with open("lin_reg.pkl", "rb") as li:  # wb: mode write
    linear_regression = pickle.load(li)

# opening pickle files (reading the models)
with open("log_reg.pkl", "rb") as lo:
    logistic_regression = pickle.load(lo)

with open("svc_.pkl", "rb") as sv:
    support_vector_classifier = pickle.load(sv)

# setting theme

# Define main function
def main():
    #title
    st.title("Iris Type Detection")
    st.subheader("**:violet[Predicting different types with 3 models]**")
    st.sidebar.header("User Input Parameters")

    # function for the User to input parameters in the sidebar
    def user_input():
        sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 6.0)
        sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.0)
        #petal_length = st.sidebar.slider("Petal Length", 1.0, 1.9, 1.5)
        #petal_width = st.sidebar.slider("Petal Width", 0.1, 0.6, 0.3)
        petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 4.0)
        petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)
        data = {"sepal_length":sepal_length,
                "sepal_width":sepal_width,
                "petal_length":petal_length,
                "petal_width":petal_width}
        features_df = pd.DataFrame(data, index=[0])
        return features_df

    df = user_input()

    # Functions to classify types of iris
    def classify_type(target):
        if target == 0:
            image = st.image("https://www.aquaplante.fr/60986-thickbox_default/iris-setosa-pot-de-9cm.jpg")
            name = "Setosa"
        elif target == 1:
            image = st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg")
            name = "Versicolor"
        else:
            image = st.image("https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_virginica_lg.jpg")
            name = "Virginica"
        return name
    # User can choose the model they wish to use
    option = {'Linear Regression', 'Logistic Regression', 'SVM Classifier'}
    model = st.sidebar.selectbox('Select model to use', option)

    # to show the measurement df
    st.write('**Your measurement input:**', df)
    st.write('**Model Chosen:**')
    st.write(model)

    # create button for running the model
    if st.button("**:violet[RUN]**"):
        if model == "Linear Regression":
            st.success(classify_type(lin_reg_fit.predict(df)[0].round()))
        elif model == "Logistic Regression":
            st.success(classify_type(log_reg_fit.predict(df)))
        else:
            st.success(classify_type(svc_fit.predict(df)))


# to run the main function !!!VERY IMPORTANT!!!
if __name__ == '__main__':
    main()
