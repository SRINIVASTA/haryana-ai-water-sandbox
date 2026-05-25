import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.stats import linregress

# --- PAGE CONFIG ---
st.set_page_config(page_title="Haryana AI Water Sandbox", layout="wide")

st.title("🌊 Haryana Groundwater Risk AI Sandbox")
st.markdown("### Prototype: NASA GRACE-FO Satellite Data Integration")
st.info("Innovator: Srinivasta | Tech: Python, xarray, Folium | Data: NASA JPL Mascon RL06.3")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Ensure this file is uploaded to your GitHub repo
    ds = xr.open_dataset('haryana_groundwater_pilot.nc')
    # Auto-identify the data variable
    var_name = [v for v in ds.data_vars if 'lwe' in v.lower() or 'thickness' in v.lower()][0]
    return ds, var_name

try:
    ds, var_name = load_data()
    hry_data = ds[var_name]

    # --- SIDEBAR METRICS ---
    st.sidebar.header("📈 State-Wide Analysis")
    
    # Calculate Burn Rate
    state_avg = hry_data.mean(dim=['lat', 'lon'])
    x = np.arange(len(state_avg))
    slope, _, _, _, _ = linregress(x, state_avg.values)
    
    st.sidebar.metric("Monthly Burn Rate", f"{slope:.4f} cm/mo", delta=f"{slope:.4f}", delta_color="inverse")
    st.sidebar.metric("Annual Depletion", f"{slope*12:.2f} cm/yr")
    
    if slope < -0.4:
        st.sidebar.error("STATUS: CRITICAL")
    else:
        st.sidebar.warning("STATUS: WATCH")

    # --- MAIN DASHBOARD ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Risk Map")
        # Initialize Folium Map
        m = folium.Map(location=[29.05, 76.08], zoom_start=8, tiles='CartoDB Positron')
        
        data_avg = hry_data.mean(dim='time')
        for lat in data_avg.lat.values:
            for lon in data_avg.lon.values:
                val = float(data_avg.sel(lat=lat, lon=lon).values)
                if not np.isnan(val):
                    # Color Logic
                    color = '#d73027' if val < -35 else '#fc8d59' if val < -15 else '#4575b4'
                    folium.Rectangle(
                        bounds=[[lat-0.25, lon-0.25], [lat+0.25, lon+0.25]],
                        color=color, fill=True, fill_opacity=0.5,
                        popup=f"Risk: {val:.2f}cm anomaly",
                        tooltip=f"Anomaly: {val:.2f}cm"
                    ).add_to(m)
        
        st_folium(m, width=800, height=500)

    with col2:
        st.subheader("East-West Risk Gap")
        # Divide at 76.2 Longitude
        west = hry_data.sel(lon=slice(74.4, 76.2)).mean(dim=['lat', 'lon'])
        east = hry_data.sel(lon=slice(76.2, 77.8)).mean(dim=['lat', 'lon'])
        
        # Calculate Zone Slopes
        s_w, _, _, _, _ = linregress(np.arange(len(west)), west.fillna(0).values)
        s_e, _, _, _, _ = linregress(np.arange(len(east)), east.fillna(0).values)
        
        st.write(f"**West Zone Burn:** {s_w:.4f} cm/mo")
        st.write(f"**East Zone Burn:** {s_e:.4f} cm/mo")
        
        gap = abs((s_e - s_w) / s_w) * 100
        st.metric("Risk Gap", f"{gap:.1f}%", help="How much faster the East is depleting compared to the West")
        
        st.markdown("---")
        st.write("**Future Scope:**")
        st.caption("1. Machine Learning Downscaling to 5km")
        st.caption("2. Real-time Observation Well Integration")
        st.caption("3. District-wise 'Day Zero' Predictions")

except Exception as e:
    st.error(f"Please upload 'haryana_groundwater_pilot.nc' to your GitHub repo. Error: {e}")

