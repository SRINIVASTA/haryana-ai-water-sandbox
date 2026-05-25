# 🌊 Haryana Groundwater Risk AI Sandbox
### AI-Driven Aquifer Monitoring using NASA Satellite Gravimetry

![Status](https://shields.io)
![Sector](https://shields.io)
![Tech](https://shields.io|%20AI%20|%20Geodata-green)

## 📌 Project Overview
This project is a technical prototype developed for the **Haryana AI Sandbox (HAIDP)** initiative. It transforms raw NASA satellite data into an actionable "Risk Rating" system for Haryana’s groundwater resources. 

By analyzing mass anomalies, we identify the **"Water Burn Rate"**—the velocity at which the state is depleting its underground water capital.

## 🚀 Key Discovery: The Risk Gap
Our analysis reveals a significant disparity between regional depletion rates:
- **East Zone (Industrial/Urban):** Highest depletion velocity due to NCR urbanization and intensive cropping.
- **West Zone (Agricultural):** Moderate depletion, buffered by surface canal systems.
- **State-Wide Burn Rate:** **-0.4638 cm/month** (~5.5 cm/year).

## 🛠️ Technical Architecture
- **Data Source:** NASA GRACE-FO (Gravity Recovery and Climate Experiment) Level-3 Mascon CRI-Filtered RL06.3.
- **Processing Engine:** Python (`xarray`, `scipy`, `numpy`) for spatial clipping and linear regression analysis.
- **Visualization:** Interactive **Folium** map integrated into a **Streamlit** dashboard.
- **Scientific Standard:** Built using methodologies learned through **IIRS/ISRO Geodata Analysis** certification.

## 📈 Future Roadmap
1. **ML Downscaling:** Utilizing Random Forest models to increase resolution from 50km to 5km using Sentinel-2 (NDVI) as a proxy.
2. **Ground-Truthing:** Ingesting in-situ data from Haryana's piezometric observation wells.
3. **Predictive Analytics:** Implementing LSTM (Long Short-Term Memory) networks to forecast "Day Zero" for all 22 districts.

## 👤 About the Innovator
**Srinivasta**
- 20+ Years in Market Risk & Analytics
- 4 Years Hands-on AI/ML Development
- Certified in Geodata Analysis (IIRS/ISRO)
- [LinkedIn Profile](https://linkedin.com) | [Portfolio App](https://streamlit.app)

---
*Developed for the Haryana AI Sandbox Launch - June 1, 2026*
