import streamlit as st
import ee
import numpy as np
import pandas as pd
from datetime import datetime
from google.oauth2 import service_account
import io
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from prophet import Prophet
import plotly.graph_objects as go
import time

@st.cache_resource
def init_ee():
    try:
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(key_dict)
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            ee.Initialize(credentials=scoped_credentials, project='satellite-454512')
            return True
        
        ee.Initialize(project='satellite-454512')
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project='satellite-454512')
            return True
        except Exception:
            return False

@st.cache_data(show_spinner=False)
def fetch_ee_data_cached(coords, start_date, end_date, scale):
    area_of_interest = ee.Geometry.Rectangle([coords[1], coords[0], coords[3], coords[2]])
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(area_of_interest)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

    def calculate_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        mean_dict = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area_of_interest,
            scale=scale,
            maxPixels=1e13
        )
        return ee.Feature(None, {
            'NDVI': mean_dict.get('NDVI'),
            'system:time_start': image.get('system:time_start')
        })

    ndvi_collection = collection.map(calculate_ndvi)
    
    try:
        info = ndvi_collection.getInfo()
    except Exception:
        return pd.DataFrame()

    features = info.get('features', [])
    dates = []
    ndvi_values = []
    for f in features:
        props = f.get('properties', {})
        val = props.get('NDVI')
        t = props.get('system:time_start')
        if val is not None and t is not None:
            dates.append(datetime.fromtimestamp(t / 1000.0))
            ndvi_values.append(val)

    df = pd.DataFrame({'Date': dates, 'NDVI': ndvi_values})
    df = df.sort_values('Date').reset_index(drop=True)
    return df

class GreenAreaAnalyzer:
    def __init__(self, locations, start_date, train_years=5, predict_years=2, flexibility=0.05, smooth_window=3, scale=250):
        self.locations = locations
        self.start_date = start_date
        self.train_years = train_years
        self.predict_years = predict_years
        self.flexibility = flexibility
        self.smooth_window = smooth_window
        self.scale = scale

    def analyze_sequential(self, start_date, end_date):
        results = {}
        for loc_name, coords in self.locations.items():
            df = fetch_ee_data_cached(tuple(coords), start_date, end_date, self.scale)
            results[loc_name] = df
        return results

    def plot_and_predict(self):
        train_start_date = self.start_date.strftime('%Y-%m-%d')
        train_end_dt = self.start_date + pd.DateOffset(years=self.train_years)
        train_end_date = train_end_dt.strftime('%Y-%m-%d')

        with st.status(f"Querying Earth Engine at {self.scale}m resolution...", expanded=True) as status:
            results = self.analyze_sequential(train_start_date, train_end_date)
            status.update(label="Data processing complete!", state="complete", expanded=False)

        for name, df in results.items():
            if df.empty:
                st.error(f"No data available for {name}.")
                continue

            df['NDVI_Smooth'] = df['NDVI'].rolling(window=self.smooth_window, min_periods=1).mean()
            
            st.markdown("<h3 class='gradient-text'>📊 Region Summary & Analytics</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            st.markdown(f"""
                <div class='glass-card'>
                    <div style='text-align:center;'>
                        <h4 style='margin:0; color:#4ade80;'>Average NDVI</h4>
                        <h2 style='margin:0;'>{df['NDVI'].mean():.4f}</h2>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            with col1:
                st.empty()
            with col2:
                st.markdown(f"""
                    <div class='glass-card'>
                        <div style='text-align:center;'>
                            <h4 style='margin:0; color:#60a5fa;'>Max NDVI</h4>
                            <h2 style='margin:0;'>{df['NDVI'].max():.4f}</h2>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class='glass-card'>
                        <div style='text-align:center;'>
                            <h4 style='margin:0; color:#f87171;'>Min NDVI</h4>
                            <h2 style='margin:0;'>{df['NDVI'].min():.4f}</h2>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class='glass-card'>
                        <div style='text-align:center;'>
                            <h4 style='margin:0; color:#a78bfa;'>Data Points</h4>
                            <h2 style='margin:0;'>{len(df)}</h2>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")

            df_prophet = df[['Date', 'NDVI_Smooth']].rename(columns={'Date': 'ds', 'NDVI_Smooth': 'y'})
            
            m = Prophet(
                changepoint_prior_scale=self.flexibility, 
                yearly_seasonality=True, 
                weekly_seasonality=False, 
                daily_seasonality=False
            )
            m.fit(df_prophet)

            future = m.make_future_dataframe(periods=self.predict_years * 12, freq='ME')
            forecast = m.predict(future)

            future_only = forecast[forecast['ds'] > train_end_date]
            if len(future_only) > 1:
                trend_diff = future_only['yhat'].iloc[-1] - future_only['yhat'].iloc[0]
                future_trend = 'Increasing 📈' if trend_diff > 0 else 'Decreasing 📉'
            else:
                future_trend = 'Stable'

            st.markdown(f"""
                <div class='glass-card' style='border-left: 4px solid #10b981;'>
                    <h4 style='margin:0;'>🌿 Prophet Forecasted Trend for <strong>{name}</strong>: <span style='color:#10b981;'>{future_trend}</span></h4>
                </div>
            """, unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📈 Live Interactive Forecast", "🔄 Seasonal Trends", "🗄️ Data Extract"])

            with tab1:
                chart_placeholder = st.empty()
                
                fig = go.Figure()
                fig.update_layout(
                    title=f'Live NDVI Forecast - {name}',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    hovermode="x unified",
                    height=600
                )

                chunk_size = max(1, len(df_prophet) // 15)
                for i in range(chunk_size, len(df_prophet) + chunk_size, chunk_size):
                    temp_df = df_prophet.iloc[:i]
                    temp_fig = go.Figure(fig)
                    temp_fig.add_trace(go.Scatter(
                        x=temp_df['ds'], y=temp_df['y'],
                        mode='markers',
                        marker=dict(color='#60a5fa', size=6, opacity=0.7),
                        name='Historical NDVI'
                    ))
                    temp_fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[forecast['ds'].min(), forecast['ds'].max()]),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[forecast['yhat_lower'].min() - 0.05, forecast['yhat_upper'].max() + 0.05])
                    )
                    chart_placeholder.plotly_chart(temp_fig, use_container_width=True)
                    time.sleep(0.05)

                final_fig = go.Figure()
                final_fig.add_trace(go.Scatter(
                    x=df_prophet['ds'], y=df_prophet['y'],
                    mode='markers',
                    marker=dict(color='#60a5fa', size=6, opacity=0.7),
                    name='Historical NDVI'
                ))
                
                final_fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                final_fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(16, 185, 129, 0.2)',
                    name='Confidence Interval'
                ))

                final_fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'],
                    mode='lines',
                    line=dict(color='#10b981', width=3),
                    name='Prophet Forecast'
                ))

                final_fig.add_vline(x=datetime.strptime(train_end_date, "%Y-%m-%d").timestamp() * 1000, line_width=2, line_dash="dash", line_color="#f87171")
                
                final_fig.update_layout(
                    title=f'NDVI Forecast - {name}',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Date"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="NDVI Value"),
                    hovermode="x unified",
                    height=600,
                    margin=dict(l=20, r=20, t=50, b=20)
                )

                chart_placeholder.plotly_chart(final_fig, use_container_width=True)

            with tab2:
                st.markdown("<h4 class='gradient-text'>Yearly Seasonality</h4>", unsafe_allow_html=True)
                
                yearly_fig = go.Figure()
                yearly_fig.add_trace(go.Scatter(
                    x=forecast['ds'][:365], 
                    y=forecast['yearly'][:365],
                    mode='lines',
                    line=dict(color='#a78bfa', width=3, shape='spline'),
                    name='Yearly Cycle'
                ))
                yearly_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat="%b"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    height=400
                )
                st.plotly_chart(yearly_fig, use_container_width=True)

                st.markdown("<h4 class='gradient-text'>Overall Trend</h4>", unsafe_allow_html=True)
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=forecast['ds'], 
                    y=forecast['trend'],
                    mode='lines',
                    line=dict(color='#f472b6', width=3),
                    name='Macro Trend'
                ))
                trend_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    height=400
                )
                st.plotly_chart(trend_fig, use_container_width=True)

            with tab3:
                export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Date', 'yhat': 'Predicted_NDVI', 'yhat_lower': 'Lower_Confidence', 'yhat_upper': 'Upper_Confidence'
                })
                export_df = pd.merge(export_df, df[['Date', 'NDVI']], on='Date', how='left')
                
                st.dataframe(export_df, use_container_width=True)
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast Dataset (CSV)",
                    data=csv_data,
                    file_name=f"{name}_prophet_data.csv",
                    mime="text/csv",
                )

def inject_custom_css():
    st.markdown("""
    <style>
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            background: rgba(255, 255, 255, 0.08);
        }
        .gradient-text {
            background: linear-gradient(90deg, #60a5fa, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3b82f6, #10b981);
            color: white;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        }
        div[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Satellite Green Area Analyzer", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
    
    inject_custom_css()
    
    col_title, col_icon = st.columns([8, 1])
    with col_title:
        st.markdown("<h1 class='gradient-text'>Satellite Green Area Analyzer</h1>", unsafe_allow_html=True)
    with col_icon:
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Copernicus_logo.svg", width=60)
    
    st.markdown("Analyze historical Sentinel-2 satellite imagery and forecast future vegetation trends with interactive live rendering.")
    st.markdown("---")
    
    if not init_ee():
        st.stop()

    st.sidebar.markdown("<h2 class='gradient-text'>📅 Timeframe & Speed</h2>", unsafe_allow_html=True)
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2019-01-01'))
    train_years = st.sidebar.slider("Training Horizon (Years)", 1, 10, 5)
    predict_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 2)
    
    scale_options = {
        500: "500m/pixel (Very Fast)",
        250: "250m/pixel (Fast)",
        100: "100m/pixel (Balanced)",
        30: "30m/pixel (Very Slow - Max Detail)"
    }
    selected_scale = st.sidebar.selectbox(
        "Processing Speed (Resolution)", 
        options=list(scale_options.keys()), 
        format_func=lambda x: scale_options[x],
        index=1
    )

    with st.sidebar.expander("⚙️ Prophet Model Settings"):
        flexibility = st.slider("Trend Flexibility", 0.001, 0.500, 0.050, 0.010)
        smooth_window = st.slider("Pre-Smoothing (Records)", 1, 10, 3)

    st.markdown("<h3 class='gradient-text'>1. Select Region</h3>", unsafe_allow_html=True)
    st.markdown("Use the **square icon** on the left side of the map to draw a rectangle over the area you want to analyze.")
    
    m = folium.Map(location=[-3.25, -59.75], zoom_start=6, tiles="CartoDB dark_matter")
    
    Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': True
        }
    ).add_to(m)

    st_data = st_folium(
        m, 
        width=1200, 
        height=400, 
        returned_objects=["all_drawings"] 
    )
    
    min_lat, max_lat, min_lon, max_lon = -3.50, -3.00, -60.00, -59.50
    loc_name = "Custom Drawn Region"
    
    if st_data.get("all_drawings"):
        coords = st_data["all_drawings"][0]["geometry"]["coordinates"][0]
        lons = [point[0] for point in coords]
        lats = [point[1] for point in coords]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        st.success(f"Region Captured! Bounds: Lat ({min_lat:.3f} to {max_lat:.3f}), Lon ({min_lon:.3f} to {max_lon:.3f})")
    else:
        st.info("No box drawn yet. The default Amazon Rainforest coordinates will be used.")

    st.markdown("---")
    st.markdown("<h3 class='gradient-text'>2. Run Prophet Forecast Analysis</h3>", unsafe_allow_html=True)

    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False

    if st.button("🚀 Analyze Selected Region", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

    if st.session_state.run_analysis:
        locations = {
            loc_name: [min_lat, min_lon, max_lat, max_lon]
        }
        analyzer = GreenAreaAnalyzer(
            locations=locations, 
            start_date=start_date, 
            train_years=train_years, 
            predict_years=predict_years,
            flexibility=flexibility,
            smooth_window=smooth_window,
            scale=selected_scale
        )
        analyzer.plot_and_predict()

if __name__ == "__main__":
    main()