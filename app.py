import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.stats import linregress

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Haryana AI Water Sandbox", layout="wide")

st.title("🌊 Haryana Groundwater Risk AI Sandbox")
st.markdown("### AI-Driven Aquifer Monitoring using NASA Satellite Gravimetry")
st.info("Innovator: Srinivasta | Tech: Python, xarray, Folium | Source: NASA JPL Mascon RL06.3")

# --- 22 DISTRICT COORDINATES ---
dist_coords = {
    "Ambala": [30.37, 76.77], "Bhiwani": [28.78, 76.13], "Charkhi Dadri": [28.59, 76.26],
    "Faridabad": [28.40, 77.31], "Fatehabad": [29.51, 75.45], "Gurugram": [28.45, 77.02],
    "Hisar": [29.14, 75.72], "Jhajjar": [28.60, 76.65], "Jind": [29.31, 76.31],
    "Kaithal": [29.80, 76.39], "Karnal": [29.68, 76.99], "Kurukshetra": [29.96, 76.84],
    "Mahendragarh": [28.26, 76.14], "Nuh": [28.11, 77.00], "Palwal": [28.14, 77.32],
    "Panchkula": [30.69, 76.86], "Panipat": [29.39, 76.96], "Rewari": [28.18, 76.61],
    "Rohtak": [28.89, 76.57], "Sirsa": [29.53, 75.02], "Sonipat": [28.99, 77.01],
    "Yamunanagar": [30.12, 77.28]
}

# --- DATA LOADING ENGINE ---
@st.cache_data
def load_data():
    ds = xr.open_dataset('haryana_groundwater_pilot.nc')
    # Target the specific variable name from your NASA processing
    target_var = '__xarray_dataarray_variable__'
    if target_var not in ds.data_vars:
        target_var = list(ds.data_vars)
    return ds, target_var

try:
    ds, target_var = load_data()
    hry_data = ds[target_var]

    # --- SIDEBAR: STATE-WIDE METRICS ---
    st.sidebar.header("📊 State-Wide Analysis")
    state_avg = hry_data.mean(dim=['lat', 'lon'])
    x = np.arange(len(state_avg))
    mask = ~np.isnan(state_avg.values)
    slope, _, _, _, _ = linregress(x[mask], state_avg.values[mask])
    
    st.sidebar.metric("Monthly Burn Rate", f"{slope:.4f} cm/mo", delta=f"{slope:.4f}", delta_color="inverse")
    st.sidebar.metric("Annual Depletion", f"{slope*12:.2f} cm/yr")
    
    if slope < -0.4:
        st.sidebar.error("STATUS: CRITICAL")
    else:
        st.sidebar.warning("STATUS: WATCH")

    # --- MAIN DASHBOARD: MAP & ANALYSIS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interactive 22-District Risk Map")
        m = folium.Map(location=[29.05, 76.08], zoom_start=8, tiles='CartoDB Positron')
        data_avg = hry_data.mean(dim='time')

        # Layer 1: NASA Data Grid
        for lat in data_avg.lat.values:
            for lon in data_avg.lon.values:
                val = float(data_avg.sel(lat=lat, lon=lon).values)
                if not np.isnan(val):
                    color = '#d73027' if val < -30 else '#fc8d59' if val < -15 else '#4575b4'
                    folium.Rectangle(
                        bounds=[[lat-0.25, lon-0.25], [lat+0.25, lon+0.25]],
                        color=color, fill=True, fill_opacity=0.3, weight=0
                    ).add_to(m)

        # Layer 2: District Clickable Markers
        for name, coords in dist_coords.items():
            dist_val = float(data_avg.sel(lat=coords, lon=coords, method='nearest').values)
            status = "⚠️ WARNING" if dist_val < -20 else "✅ STABLE"
            folium.Marker(
                location=coords,
                icon=folium.Icon(color='red' if dist_val < -20 else 'blue', icon='info-sign'),
                popup=f"<b>{name}</b><br>Anomaly: {dist_val:.2f} cm<br>Status: {status}",
                tooltip=name
            ).add_to(m)
        
        st_folium(m, width=800, height=550)

    with col2:
        st.subheader("Regional Risk Comparison")
        # Divide state at 76.2 Longitude (East-West Gap)
        west = hry_data.sel(lon=slice(74.4, 76.2)).mean(dim=['lat', 'lon'])
        east = hry_data.sel(lon=slice(76.2, 77.8)).mean(dim=['lat', 'lon'])
        
        # Calculate Zone Burn Rates
        s_w, _, _, _, _ = linregress(np.arange(len(west)), west.fillna(0).values)
        s_e, _, _, _, _ = linregress(np.arange(len(east)), east.fillna(0).values)
        
        st.write(f"🏠 **West Zone Burn:** {s_w:.4f} cm/mo")
        st.write(f"🏢 **East Zone Burn:** {s_e:.4f} cm/mo")
        
        gap = abs((s_e - s_w) / s_w) * 100
        st.metric("Risk Gap (East vs West)", f"{gap:.1f}%")
        
        st.markdown("---")
        st.markdown("### 🛠 AI Sandbox Roadmap")
        st.write("1. **Spatial Downscaling:** Increase resolution to 5km using ML.")
        st.write("2. **Local Integration:** Merge with HWRA piezometric well data.")
        st.write("3. **Day Zero Prediction:** District-wise water exhaustion dates.")

except Exception as e:
    st.error("Deployment Error")
    st.write(f"Check if 'haryana_groundwater_pilot.nc' is on GitHub. Detail: {e}")
