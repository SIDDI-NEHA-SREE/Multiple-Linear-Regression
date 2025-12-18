import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="MLR", layout="centered")


# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> using Multiple Linear Regression</p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["total_bill","size","tip"]].head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare Data
X,y = df[['total_bill','size']],df['tip']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip(MLR)")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"],color="black",alpha=0.6)

ax.plot(df["total_bill"],model.predict(scaler.transform(X)),color="red",linewidth=2)
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Performance Metrics
st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")


# m & c #
st.markdown(f"""
    <div class = "card">
    <h3>Model Interception</h3>
   <p><b>Coefficient (Total bill):</b> {model.coef_[0]:.3f}<br>
    <b>co-efficient (Group size) :</b> {model.coef_[1] :.3f}<br>
    <b>Intercept :</b> {model.intercept_:.3f}</p>
    <p>Tip Depends upon the <b>Bill amount</b> and <b> number of people </b>.
    </p>
    </div>""", unsafe_allow_html = True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    30.0
)
size=st.slider("Group Size",
               int(df["size"].min()),
                   int(df["size"].max()),2)
input_scaled = scaler.transform([[bill,size]])
tip = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
