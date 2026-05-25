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
st.markdown("### Satellite-Driven Aquifer Monitoring & Risk Assessment")
st.info("Innovator: Srinivasta | Source: NASA JPL Mascon CRI-Filtered | Tech: Python & AI")

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

@st.cache_data
def load_data():
    try:
        ds = xr.open_dataset('haryana_groundwater_pilot.nc')
        # Identify the data variable (handling the generic name from Colab)
        valid_vars = [v for v in ds.data_vars if v not in ['lat', 'lon', 'time', 'scale_factor']]
        if not valid_vars:
            valid_vars = [list(ds.data_vars)[0]]
        return ds, valid_vars[0]
    except Exception as e:
        return None, str(e)

ds, target_var = load_data()

if ds is None:
    st.error(f"⚠️ Deployment Error: {target_var}")
else:
    hry_data = ds[target_var]
    
    # --- SIDEBAR: ANALYSIS & ROADMAP ---
    st.sidebar.header("📊 State-Wide Analysis")
    state_avg = hry_data.mean(dim=['lat', 'lon']).values.flatten()
    x_axis = np.arange(len(state_avg))
    mask = ~np.isnan(state_avg)
    slope, _, _, _, _ = linregress(x_axis[mask], state_avg[mask])
    
    st.sidebar.metric("Monthly Burn Rate", f"{slope:.4f} cm/mo", delta=f"{slope:.4f}", delta_color="inverse")
    st.sidebar.metric("Annual Depletion", f"{slope*12:.2f} cm/yr")

    # REGIONAL RISK GAP
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Regional Risk Gap")
    west = hry_data.sel(lon=slice(74.4, 76.2)).mean(dim=['lat', 'lon']).values.flatten()
    east = hry_data.sel(lon=slice(76.2, 77.8)).mean(dim=['lat', 'lon']).values.flatten()
    
    s_w, _, _, _, _ = linregress(np.arange(len(west)), np.nan_to_num(west))
    s_e, _, _, _, _ = linregress(np.arange(len(east)), np.nan_to_num(east))
    
    st.sidebar.write(f"🏠 **West Zone Burn:** {s_w:.4f} cm/mo")
    st.sidebar.write(f"🏢 **East Zone Burn:** {s_e:.4f} cm/mo")
    
    gap = abs((s_e - s_w) / s_w) * 100
    st.sidebar.metric("Risk Gap (East vs West)", f"{gap:.1f}%")

    # AI SANDBOX ROADMAP
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 AI Sandbox Roadmap")
    st.sidebar.write("1. **Spatial Downscaling:** Increase resolution to 5km via Random Forest.")
    st.sidebar.write("2. **Ground Truth:** Merge with Haryana Govt (HWRA) well data.")
    st.sidebar.write("3. **Prediction:** Forecast water exhaustion dates for 22 districts.")

    # --- MAIN DASHBOARD: FULL WIDTH MAP ---
    st.subheader("Interactive 22-District Risk Map")
    m = folium.Map(location=[29.05, 76.08], zoom_start=8, tiles='CartoDB Positron')
    data_avg = hry_data.mean(dim='time')

    # Layer 1: Data Grid Background
    for lat in data_avg.lat.values:
        for lon in data_avg.lon.values:
            val = data_avg.sel(lat=lat, lon=lon).values.item()
            if not np.isnan(val):
                color = '#d73027' if val < -30 else '#fc8d59' if val < -15 else '#4575b4'
                folium.Rectangle(
                    bounds=[[lat-0.25, lon-0.25], [lat+0.25, lon+0.25]],
                    color=color, fill=True, fill_opacity=0.3, weight=0
                ).add_to(m)

    # Layer 2: District Markers with Hover & Popup
    for name, coords in dist_coords.items():
        dist_val = data_avg.sel(lat=coords, lon=coords, method='nearest').values.item()
        
        if dist_val < -30:
            status, icon_color = "CRITICAL 🚨", 'darkred'
        elif dist_val < -15:
            status, icon_color = "WARNING ⚠️", 'orange'
        else:
            status, icon_color = "STABLE ✅", 'blue'

        folium.Marker(
            location=coords,
            icon=folium.Icon(color=icon_color, icon='info-sign'),
            popup=f"<b>{name}</b><br>Status: {status}<br>Anomaly: {dist_val:.2f} cm",
            tooltip=f"{name} | {status} ({dist_val:.2f} cm)"
        ).add_to(m)
    
    st_folium(m, width=1300, height=700)
