import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def Linearregression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='black', label='Data Points')
    ax.plot(X, y_pred, color='blue', linewidth=3, label='Linear Regression')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression')
    ax.legend()
    st.pyplot(fig)

def lasso(X, y, alpha):
    lasso_model = Lasso(alpha=alpha)  
    lasso_model.fit(X, y)
    y_pred_lasso = lasso_model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='black', label='Data Points')
    ax.plot(X, y_pred_lasso, color='red', linewidth=2, label='Lasso Regression')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Lasso Regression')
    ax.legend()
    st.pyplot(fig)

def ridge(X, y, alpha):
    ridge_model = Ridge(alpha=alpha)  
    ridge_model.fit(X, y)
    y_pred_ridge = ridge_model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='black', label='Data Points')
    ax.plot(X, y_pred_ridge, color='blue', linewidth=2, label='Ridge Regression')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Ridge Regression')
    ax.legend()
    st.pyplot(fig)

np.random.seed(42)  # Set seed for reproducibility
X = np.linspace(1, 10, 200).reshape(-1, 1)  # Generate 200 evenly spaced values between 1 and 10
y = 2 * X.squeeze() + np.random.normal(0, 2, 200)

st.sidebar.header('Regression Type')
regression_type = st.sidebar.selectbox('Select Regression Type', ('Linear Regression', 'Lasso Regression', 'Ridge Regression'))

if regression_type == 'Linear Regression':
    Linearregression(X, y)
elif regression_type == 'Lasso Regression':
    alpha = st.sidebar.slider('Select Alpha', min_value=0.1, max_value=10.0, step=0.1)
    lasso(X, y, alpha)
else: # Ridge Regression
    alpha = st.sidebar.slider('Select Alpha', min_value=0.1, max_value=10.0, step=0.1)
    ridge(X, y, alpha)
