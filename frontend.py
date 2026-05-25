import streamlit as st

# Import your existing modules
# We use aliases to avoid naming conflicts since they share function names
import MLP as mlp_backend
import RNN as rnn_backend
import NB as nb_backend

st.set_page_config(page_title="Crop Rotation Optimizer", layout="wide")

st.title("🌾 Crop Rotation Optimization Dashboard")
st.markdown("Use Digital Twins (MLP / RNN / Naive Bayes) to find the optimal crop sequence for your paddock.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Configuration")
engine_choice = st.sidebar.selectbox(
    "Select Prediction Engine:",
    ["MLP Surrogate", "RNN Surrogate", "Naive Bayes Classifier"]
)

st.sidebar.markdown("---")
st.sidebar.header("Location Input")
lat_input = st.sidebar.number_input("Latitude", value=-33.8, format="%.4f")
lon_input = st.sidebar.number_input("Longitude", value=151.2, format="%.4f")

# Caching the training step so it only happens ONCE while the app is running
@st.cache_resource
def load_mlp():
    return mlp_backend.load_or_train_model()

@st.cache_resource
def load_rnn():
    return rnn_backend.load_or_train_model()

@st.cache_resource
def load_nb():
    # Load NB model (trains on startup)
    return nb_backend

# --- MAIN DASHBOARD ---
st.write(f"### Currently using: **{engine_choice}**")

if st.sidebar.button("Run Optimizer / Simulator", type="primary"):
    
    with st.spinner("Processing Data..."):
        try:
            if engine_choice == "MLP Surrogate":
                # 1. Load/Train Model
                st.toast("Loading MLP Model... Please wait.", icon="⏳")
                trained_model, scaler_x, scaler_y = load_mlp()
                
                # 2. Fetch Enviro Data
                enviro_data = mlp_backend.get_enviro_data(lat_input, lon_input)
                st.info(f"📍 Matched Location Data: Rainfall={enviro_data[0]:.1f}mm, Elevation={enviro_data[1]:.1f}m")
                
                # 3. Optimize
                best_seq, expected_revenue, best_yields = mlp_backend.optimize_rotation(
                    trained_model, scaler_x, scaler_y, enviro_data, iterations=2000
                )
                
                st.success("Optimization Complete!")
                st.metric(label="Optimal 5-Year Rotation", value=best_seq)
                st.metric(label="Predicted Revenue", value=f"${expected_revenue:.2f} / ha")
                
                st.write("**Predicted Yearly Yields:**")
                cols = st.columns(5)
                crops = [c.strip() for c in best_seq.split(",")]
                for year, (crop, yield_val) in enumerate(zip(crops, best_yields)):
                    if year < 5:
                        cols[year].metric(label=f"Year {year+1} ({crop})", value=f"{yield_val:.1f} kg/ha")

            elif engine_choice == "RNN Surrogate":
                # 1. Load/Train Model
                st.toast("Loading RNN Model... Please wait.", icon="⏳")
                trained_model, scaler_x, scaler_y = load_rnn()
                
                # 2. Fetch Enviro Data
                enviro_data = rnn_backend.get_enviro_data(lat_input, lon_input)
                st.info(f"📍 Matched Location Data: Rainfall={enviro_data[0]:.1f}mm, N={enviro_data[6]:.1f}kg/ha")
                
                # 3. Optimize
                best_seq, expected_revenue, best_yields = rnn_backend.optimize_rotation(
                    trained_model, scaler_x, scaler_y, enviro_data, iterations=2000
                )
                
                st.success("Optimization Complete!")
                st.metric(label="Optimal 5-Year Rotation", value=best_seq)
                st.metric(label="Predicted Revenue", value=f"${expected_revenue:.2f} / ha")
                
                st.write("**Predicted Yearly Yields:**")
                cols = st.columns(5)
                crops = [c.strip() for c in best_seq.split(",")]
                for year, (crop, yield_val) in enumerate(zip(crops, best_yields)):
                    if year < 5:
                        cols[year].metric(label=f"Year {year+1} ({crop})", value=f"{yield_val:.1f} kg/ha")

            elif engine_choice == "Naive Bayes Classifier":
                # 1. Load NB Model
                st.toast("Loading Naive Bayes Model... Please wait.", icon="⏳")
                nb_model = load_nb()
                
                # 2. Predict rotation using coordinates
                st.info(f"📍 Finding nearby location with similar soil properties...")
                predicted_rotation = nb_model.predict_crop_rotation_by_coordinates(lat_input, lon_input)
                
                st.success("Prediction Complete!")
                st.metric(label="Recommended Crop Rotation", value=predicted_rotation)
                st.write(f"*Based on soil properties and environmental conditions at ({lat_input:.4f}, {lon_input:.4f})*")

        except Exception as e:
            st.error(f"An error occurred: {e}")
