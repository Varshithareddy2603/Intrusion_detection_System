import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label"
    ]
    df = pd.read_csv(url, names=columns)
    return df

df = load_data()

# Preprocessing
def preprocess_data(df):
    df = df.dropna()

    label_encoders = {}
    categorical_columns = ["protocol_type", "service", "flag"]
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    attack_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

    X = df.drop(columns=["label"])
    y = df["label"]

    X = X.apply(pd.to_numeric, errors='coerce')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders, X.columns, label_encoder, attack_mapping

X, y, scaler, label_encoders, feature_names, label_encoder, attack_mapping = preprocess_data(df)

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"**Model Accuracy:** {acc:.2f}")
    
    return model

model = train_model(X, y)

# Streamlit UI
st.title("Intrusion Detection System (IDS)")
st.sidebar.header("Enter Network Traffic Features")

user_input = {}

for col in feature_names:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        user_input[col] = st.sidebar.selectbox(col, options)
    else:
        user_input[col] = st.sidebar.number_input(col, value=0.0)

for col in label_encoders:
    user_input[col] = label_encoders[col].transform([user_input[col]])[0]

input_df = pd.DataFrame([user_input])
input_df = input_df[feature_names]
input_array = input_df.to_numpy()
input_scaled = scaler.transform(input_array)

if st.sidebar.button("Detect Intrusion"):
    predicted_label = model.predict(input_scaled)[0]
    attack_name = attack_mapping[predicted_label]
    
    if attack_name == "normal":
        st.success("✅ Normal Traffic")
    else:
        st.error(f"⚠ Intrusion Detected! Attack Type: **{attack_name}**")

        



