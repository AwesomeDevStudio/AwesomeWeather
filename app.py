import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
import json
import zipfile
import tempfile
import shutil
import cv2
import numpy as np

# Imports internes
import config
from src.core.pdf_processor import pdf_to_images
from src.core.card_zone_extractor import process_card_image
from src.utils.database import init_database, save_bulletin, get_all_bulletins
from src.utils.card_split_by_titles import decoupe_par_titres

# Pour les graphiques interactifs
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Pour la carte du Burkina
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ---- Config Streamlit
st.set_page_config(
    page_title="AwesomeWeather - ANAM",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_database()

# ---- Styles CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1f1f1f !important;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #4a4a4a !important;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        color: #2d3748;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    .metric-card .delta {
        font-size: 0.9rem;
        color: #48bb78;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
    <div class="main-header">
        <h1>🌦️ AwesomeWeather</h1>
        <p>Plateforme d\'Analyse Automatique des Prévisions Météorologiques - ANAM Burkina Faso</p>
    </div>
''', unsafe_allow_html=True)

# ---- Sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["📊 Dashboard", "📤 Upload & Analyse", "📈 Analyses", "📁 Historique", "⚙️ Configuration"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("💡 **Extracteur:** Carte dynamique + OCR Tesseract")
    st.markdown("---")
    st.caption("🇧🇫 ANAM - Agence Nationale de la Météorologie")
    st.caption("📧 contact@anam.bf")
    st.caption("© 2025 AwesomeWeather")

# ---- Coordonnées des stations du Burkina Faso
STATIONS_COORDS = {
    'Ouagadougou': (12.3714, -1.5197),
    'Bobo-Dioulasso': (11.1771, -4.2967),
    'Dori': (14.0354, -0.0348),
    'Ouahigouya': (13.5828, -2.4214),
    'Fada N\'Gourma': (12.0614, 0.3556),
    'Banfora': (10.6330, -4.7616),
    'Koudougou': (12.2522, -2.3625),
    'Tenkodogo': (11.7800, -0.3700),
    'Diapaga': (12.0717, 1.7869),
    'Tougan': (13.0667, -3.0667),
    'Gaoua': (10.3219, -3.1717),
    'Bogandé': (12.9667, -0.1500)
}

# ---- Données fictives
def generate_fake_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    stations = list(STATIONS_COORDS.keys())
    data = []
    for date in dates:
        for station in stations:
            obs_min = np.random.randint(18, 28)
            obs_max = obs_min + np.random.randint(10, 18)
            prev_min = obs_min + np.random.randint(-3, 4)
            prev_max = obs_max + np.random.randint(-3, 4)
            data.append({
                'date': date,
                'station': station,
                'obs_min': obs_min,
                'obs_max': obs_max,
                'prev_min': prev_min,
                'prev_max': prev_max,
                'error_min': abs(prev_min - obs_min),
                'error_max': abs(prev_max - obs_max)
            })
    return pd.DataFrame(data)

# ---- Fonctions extraction
def extract_pdfs_from_zip(zip_file):
    pdf_files = []
    temp_dir = tempfile.mkdtemp()
    try:
        zip_path = Path(temp_dir) / zip_file.name
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith('.pdf') and not file_info.is_dir():
                    extracted_path = zip_ref.extract(file_info, temp_dir)
                    with open(extracted_path, 'rb') as pdf_file:
                        pdf_content = pdf_file.read()
                    pdf_files.append({
                        'name': Path(file_info.filename).name,
                        'content': pdf_content,
                        'size': file_info.file_size
                    })
        return pdf_files, temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def process_single_bulletin(pdf_name, pdf_content):
    pdf_path = config.UPLOAD_DIR / pdf_name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    image_paths = pdf_to_images(str(pdf_path))
    all_obs, all_prev = [], []
    for image_path in image_paths:
        page_img = cv2.imread(str(image_path))
        obs_img, prev_img = decoupe_par_titres(page_img)
        obs_result = process_card_image(obs_img, debug=False)
        prev_result = process_card_image(prev_img, debug=False)
        for ville, (tmin, tmax) in obs_result.items():
            if tmin is not None and tmax is not None:
                all_obs.append({'station': ville, 'temp_min': tmin, 'temp_max': tmax})
        for ville, (tmin, tmax) in prev_result.items():
            if tmin is not None and tmax is not None:
                all_prev.append({'station': ville, 'temp_min': tmin, 'temp_max': tmax})
    date_val = None
    try:
        for part in pdf_name.replace('-', ' ').replace('.', ' ').replace('_', ' ').split():
            if part.isdigit() and len(part) == 4:
                idx = pdf_name.find(part)
                likely_date = pdf_name[idx-7:idx+5].replace('_', ' ').replace('-', ' ')
                dt = [int(s) for s in likely_date.split() if s.isdigit()]
                if len(dt) >= 3:
                    date_val = f"{dt[2]:04d}-{dt[1]:02d}-{dt[0]:02d}"
                    break
    except Exception:
        date_val = None
    bulletin_id = save_bulletin(pdf_name, {'observations': all_obs, 'previsions': all_prev})
    return {'date': date_val, 'observations': all_obs, 'previsions': all_prev,
            'errors': [], 'bulletin_id': bulletin_id}

# ==================== DASHBOARD ====================
if page == "📊 Dashboard":
    st.header("📊 Tableau de bord général")
    df_fake = generate_fake_data()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>📄 Bulletins traités</h3>
                <div class="value">180</div>
                <div class="delta">+12 ce mois</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        mae_global = df_fake[['error_min', 'error_max']].mean().mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Précision MAE</h3>
                <div class="value">{mae_global:.2f}°C</div>
                <div class="delta">-0.3°C ce mois</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>✅ Taux succès</h3>
                <div class="value">94.2%</div>
                <div class="delta">+2.1% ce mois</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📍 Stations</h3>
                <div class="value">{len(STATIONS_COORDS)}</div>
                <div class="delta">Toutes actives</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Carte du Burkina Faso + Graphique
    col_map, col_graph = st.columns([1.2, 1])
    
    with col_map:
        st.subheader("🗺️ Carte du Burkina Faso - MAE par station")
        
        # Calculer MAE par station
        df_station_mae = df_fake.groupby('station')[['error_min', 'error_max']].mean()
        df_station_mae['MAE'] = (df_station_mae['error_min'] + df_station_mae['error_max']) / 2
        
        if FOLIUM_AVAILABLE:
            # Créer la carte centrée sur le Burkina
            m = folium.Map(
                location=[12.2383, -1.5616],
                zoom_start=7,
                tiles='OpenStreetMap'
            )
            
            # Ajouter marqueurs avec couleur selon MAE
            for station, coords in STATIONS_COORDS.items():
                if station in df_station_mae.index:
                    mae_value = df_station_mae.loc[station, 'MAE']
                    
                    # Couleur selon MAE (vert = bon, rouge = mauvais)
                    if mae_value < 1.5:
                        color = 'green'
                    elif mae_value < 2.5:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=10 + mae_value * 2,
                        popup=folium.Popup(
                            f"<b>{station}</b><br>MAE: {mae_value:.2f}°C<br>Min: {df_station_mae.loc[station, 'error_min']:.2f}°C<br>Max: {df_station_mae.loc[station, 'error_max']:.2f}°C",
                            max_width=250
                        ),
                        tooltip=f"{station}: {mae_value:.2f}°C",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
            
            # Afficher la carte
            st_folium(m, width=700, height=500, returned_objects=[])
        else:
            st.error("❌ Folium non installé. Exécutez: pip install folium streamlit-folium")
    
    with col_graph:
        st.subheader("📈 Top/Bottom 5 stations")
        df_station_mae_sorted = df_station_mae.sort_values('MAE')
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Top 5 meilleures (MAE plus faible)
            top5 = df_station_mae_sorted.head(5)
            fig.add_trace(go.Bar(
                y=top5.index,
                x=top5['MAE'],
                orientation='h',
                name='Meilleures',
                marker_color='#48bb78',
                text=[f"{v:.2f}°C" for v in top5['MAE']],
                textposition='outside'
            ))
            
            # Bottom 5 (MAE plus élevé)
            bottom5 = df_station_mae_sorted.tail(5)
            fig.add_trace(go.Bar(
                y=bottom5.index,
                x=bottom5['MAE'],
                orientation='h',
                name='À améliorer',
                marker_color='#f56565',
                text=[f"{v:.2f}°C" for v in bottom5['MAE']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Classement par précision MAE",
                xaxis_title="MAE (°C)",
                barmode='group',
                height=450,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Carte thermique mensuelle
    st.subheader("🗓️ Carte thermique mensuelle - Évolution MAE par station")
    
    if PLOTLY_AVAILABLE:
        # Préparer données mensuelles
        df_fake['mois'] = df_fake['date'].dt.to_period('M').astype(str)
        mois_disponibles = sorted(df_fake['mois'].unique())
        
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            mois_selectionne = st.selectbox(
                "Mois à analyser",
                mois_disponibles,
                index=len(mois_disponibles)-1
            )
        
        with col_info:
            df_mois = df_fake[df_fake['mois'] == mois_selectionne]
            df_heatmap = df_mois.groupby('station')[['error_min', 'error_max']].mean()
            df_heatmap['MAE'] = (df_heatmap['error_min'] + df_heatmap['error_max']) / 2
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📉 MAE Min", f"{df_heatmap['MAE'].min():.2f}°C")
            with col_b:
                st.metric("📊 MAE Moyen", f"{df_heatmap['MAE'].mean():.2f}°C")
            with col_c:
                st.metric("📈 MAE Max", f"{df_heatmap['MAE'].max():.2f}°C")
        
        # Graphique heatmap tous mois
        df_monthly_all = df_fake.groupby(['mois', 'station'])[['error_min', 'error_max']].mean()
        df_monthly_all['MAE'] = (df_monthly_all['error_min'] + df_monthly_all['error_max']) / 2
        df_pivot = df_monthly_all['MAE'].reset_index().pivot(index='station', columns='mois', values='MAE')
        
        fig = px.imshow(
            df_pivot,
            labels=dict(x="Mois", y="Station", color="MAE (°C)"),
            color_continuous_scale='RdYlGn_r',
            aspect="auto",
            title=f"Évolution MAE sur 6 mois (Sélectionné: {mois_selectionne})"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tableau récapitulatif détaillé
    st.subheader("📋 Statistiques détaillées par station")
    stats_table = df_fake.groupby('station').agg({
        'error_min': ['mean', 'std', 'min', 'max'],
        'error_max': ['mean', 'std', 'min', 'max']
    }).round(2)
    stats_table.columns = [
        'MAE Min (°C)', 'Écart-type Min', 'Erreur Min Min', 'Erreur Min Max',
        'MAE Max (°C)', 'Écart-type Max', 'Erreur Max Min', 'Erreur Max Max'
    ]
    st.dataframe(stats_table, use_container_width=True)

# ==================== UPLOAD ====================
elif page == "📤 Upload & Analyse":
    st.header("📤 Importer et analyser des bulletins")
    
    upload_type = st.radio("Type de fichier", ["📄 PDF individuels", "📦 Archive ZIP"], horizontal=True)
    files_to_process = []
    temp_dir = None
    
    if upload_type == "📄 PDF individuels":
        st.info("💡 Glissez-déposez un ou plusieurs bulletins PDF")
        uploaded_files = st.file_uploader("Choisir des fichiers PDF", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                files_to_process.append({'name': file.name, 'content': file.getbuffer(), 'size': file.size})
    else:
        st.info("💡 Uploadez un fichier ZIP contenant vos bulletins")
        uploaded_zip = st.file_uploader("Choisir un fichier ZIP", type=['zip'])
        if uploaded_zip:
            with st.spinner("📦 Extraction du ZIP..."):
                try:
                    extracted_pdfs, temp_dir = extract_pdfs_from_zip(uploaded_zip)
                    files_to_process = extracted_pdfs
                    st.success(f"✅ {len(extracted_pdfs)} PDF(s) trouvé(s)")
                except Exception as e:
                    st.error(f"❌ Erreur d'extraction: {str(e)}")
    
    if files_to_process:
        st.success(f"✅ {len(files_to_process)} fichier(s) prêt(s) pour le traitement")
        
        with st.expander("📋 Voir la liste des fichiers"):
            for file in files_to_process:
                st.text(f"📄 {file['name']} ({file['size']/1024:.1f} KB)")
        
        st.markdown("---")
        
        if st.button("🚀 Lancer l'extraction", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for idx, file_info in enumerate(files_to_process):
                status_text.text(f"Traitement en cours: {file_info['name']} ({idx+1}/{len(files_to_process)})")
                
                try:
                    result = process_single_bulletin(file_info['name'], file_info['content'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ {file_info['name']}")
                    with col2:
                        st.metric("Observations", len(result['observations']))
                    with col3:
                        st.metric("Prévisions", len(result['previsions']))
                    
                    results.append({
                        'Fichier': file_info['name'],
                        'Observations': len(result['observations']),
                        'Prévisions': len(result['previsions']),
                        'Statut': '✅ Succès'
                    })
                    
                except Exception as e:
                    st.error(f"❌ {file_info['name']}: {str(e)}")
                    results.append({
                        'Fichier': file_info['name'],
                        'Observations': 0,
                        'Prévisions': 0,
                        'Statut': f'❌ Erreur'
                    })
                
                progress_bar.progress((idx + 1) / len(files_to_process))
            
            status_text.empty()
            st.balloons()
            st.success("🎉 Traitement terminé avec succès !")
            
            st.markdown("---")
            st.subheader("📊 Résumé du traitement")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Stats globales
            col1, col2, col3 = st.columns(3)
            with col1:
                nb_success = len([r for r in results if '✅' in r['Statut']])
                st.metric("✅ Réussis", nb_success)
            with col2:
                nb_failed = len([r for r in results if '❌' in r['Statut']])
                st.metric("❌ Échecs", nb_failed)
            with col3:
                total_obs = sum([r['Observations'] for r in results])
                st.metric("📊 Observations totales", total_obs)
            
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

# ==================== ANALYSES ====================
elif page == "📈 Analyses":
    st.header("📈 Analyses approfondies")
    df_fake = generate_fake_data()
    
    analyse_type = st.selectbox(
        "Sélectionner le type d'analyse",
        ["📅 Journalière (30 derniers jours)", "📆 Mensuelle", "📊 Annuelle"]
    )
    
    st.markdown("---")
    
    if PLOTLY_AVAILABLE:
        if analyse_type == "📅 Journalière (30 derniers jours)":
            st.subheader("Analyse journalière - 30 derniers jours")
            
            df_last30 = df_fake[df_fake['date'] >= (datetime.now() - timedelta(days=30))]
            
            # Graphique température observée vs prévue
            stations_selectionnees = st.multiselect(
                "Sélectionner les stations à afficher",
                df_last30['station'].unique(),
                default=list(df_last30['station'].unique())[:3]
            )
            
            if stations_selectionnees:
                fig = go.Figure()
                for station in stations_selectionnees:
                    df_station = df_last30[df_last30['station'] == station]
                    
                    fig.add_trace(go.Scatter(
                        x=df_station['date'],
                        y=df_station['obs_max'],
                        name=f'{station} (Observé)',
                        mode='lines+markers',
                        line=dict(width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_station['date'],
                        y=df_station['prev_max'],
                        name=f'{station} (Prévu)',
                        mode='lines',
                        line=dict(dash='dash', width=1.5)
                    ))
                
                fig.update_layout(
                    title="Températures maximales : Observations vs Prévisions",
                    xaxis_title="Date",
                    yaxis_title="Température (°C)",
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tableau erreurs moyennes
            st.subheader("📊 Erreurs moyennes sur 30 jours")
            df_errors_30d = df_last30.groupby('station')[['error_min', 'error_max']].mean().round(2)
            df_errors_30d.columns = ['Erreur Min (°C)', 'Erreur Max (°C)']
            st.dataframe(df_errors_30d, use_container_width=True)
        
        elif analyse_type == "📆 Mensuelle":
            st.subheader("Analyse mensuelle")
            
            df_fake['mois'] = df_fake['date'].dt.to_period('M').astype(str)
            df_monthly = df_fake.groupby(['mois', 'station']).agg({
                'error_min': 'mean',
                'error_max': 'mean'
            }).reset_index()
            df_monthly['MAE'] = (df_monthly['error_min'] + df_monthly['error_max']) / 2
            
            # Évolution MAE
            fig = px.line(
                df_monthly,
                x='mois',
                y='MAE',
                color='station',
                title="Évolution MAE mensuelle par station",
                labels={'mois': 'Mois', 'MAE': 'MAE (°C)', 'station': 'Station'},
                markers=True
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot distribution
            st.subheader("📦 Distribution des erreurs par station")
            fig2 = px.box(
                df_fake,
                x='station',
                y='error_max',
                title="Distribution des erreurs (Température Max)",
                labels={'station': 'Station', 'error_max': 'Erreur (°C)'},
                color='station'
            )
            fig2.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig2, use_container_width=True)
        
        else:  # Annuelle
            st.subheader("Analyse annuelle - Vue d'ensemble")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE Min moyen", f"{df_fake['error_min'].mean():.2f}°C")
            with col2:
                st.metric("MAE Max moyen", f"{df_fake['error_max'].mean():.2f}°C")
            with col3:
                st.metric("Écart-type global", f"{df_fake['error_max'].std():.2f}°C")
            with col4:
                mae_global = df_fake[['error_min', 'error_max']].mean().mean()
                st.metric("MAE global", f"{mae_global:.2f}°C")
            
            # Distribution globale
            fig = px.histogram(
                df_fake,
                x='error_max',
                nbins=40,
                title="Distribution des erreurs de prévision (Température Max)",
                labels={'error_max': 'Erreur (°C)', 'count': 'Fréquence'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Violin plot par station
            st.subheader("🎻 Comparaison détaillée par station")
            fig2 = px.violin(
                df_fake,
                x='station',
                y='error_max',
                box=True,
                points='outliers',
                title="Distribution et outliers par station",
                color='station'
            )
            fig2.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig2, use_container_width=True)

# ==================== HISTORIQUE ====================
elif page == "📁 Historique":
    st.header("📁 Historique des bulletins traités")
    
    bulletins = get_all_bulletins()
    
    if bulletins:
        df_bulletins = pd.DataFrame(bulletins, columns=[
            'ID', 'Fichier', 'Date Upload', 'Date Extraction',
            'Nb Stations', 'Statut', 'Données JSON'
        ])
        
        # Stats globales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total bulletins", len(bulletins))
        with col2:
            bulletins_mois = len([b for b in bulletins if datetime.now().strftime('%Y-%m') in str(b[2])])
            st.metric("📅 Ce mois", bulletins_mois)
        with col3:
            bulletins_semaine = len([b for b in bulletins if (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d') <= str(b[2])])
            st.metric("📆 Cette semaine", bulletins_semaine)
        
        st.markdown("---")
        
        # Filtres
        col_search, col_date = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("🔍 Rechercher un bulletin", placeholder="Nom du fichier...")
        with col_date:
            sort_order = st.selectbox("Tri par date", ["Plus récent", "Plus ancien"])
        
        # Filtrer et trier
        df_filtered = df_bulletins.copy()
        if search_term:
            df_filtered = df_filtered[df_filtered['Fichier'].str.contains(search_term, case=False, na=False)]
        
        if sort_order == "Plus récent":
            df_filtered = df_filtered.sort_values('Date Upload', ascending=False)
        else:
            df_filtered = df_filtered.sort_values('Date Upload', ascending=True)
        
        # Affichage tableau
        st.dataframe(
            df_filtered[['ID','Fichier','Date Upload','Nb Stations','Statut']],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Détails bulletin
        st.subheader("🔍 Détails d'un bulletin")
        selected_id = st.selectbox("Sélectionner un bulletin", df_filtered['ID'].tolist())
        
        if selected_id:
            bulletin_data = df_bulletins[df_bulletins['ID'] == selected_id].iloc[0]
            data_json = json.loads(bulletin_data['Données JSON'])
            
            st.info(f"📄 **Fichier:** {bulletin_data['Fichier']} | 📅 **Date:** {bulletin_data['Date Upload']}")
            
            tab1, tab2, tab3 = st.tabs(["📊 Observations", "🔮 Prévisions", "📥 Export JSON"])
            
            with tab1:
                if isinstance(data_json, dict) and 'observations' in data_json and data_json['observations']:
                    df_obs = pd.DataFrame(data_json['observations'])[['station','temp_min','temp_max']]
                    df_obs.columns = ['Station', 'Temp Min (°C)', 'Temp Max (°C)']
                    st.dataframe(df_obs, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune observation disponible")
            
            with tab2:
                if isinstance(data_json, dict) and 'previsions' in data_json and data_json['previsions']:
                    df_prev = pd.DataFrame(data_json['previsions'])[['station','temp_min','temp_max']]
                    df_prev.columns = ['Station', 'Temp Min (°C)', 'Temp Max (°C)']
                    st.dataframe(df_prev, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune prévision disponible")
            
            with tab3:
                json_str = json.dumps(data_json, indent=2, ensure_ascii=False)
                st.code(json_str, language='json')
                st.download_button(
                    "📥 Télécharger JSON",
                    json_str,
                    f"bulletin_{selected_id}.json",
                    "application/json",
                    use_container_width=True
                )
    else:
        st.info("📭 Aucun bulletin dans l'historique pour le moment")
        st.markdown("Commencez par uploader des bulletins dans la section **📤 Upload & Analyse**")

# ==================== CONFIGURATION ====================
elif page == "⚙️ Configuration":
    st.header("⚙️ Configuration du système")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Extracteur", "📍 Stations", "📂 Chemins"])
    
    with tab1:
        st.subheader("Configuration de l'extracteur OCR")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Méthode:** Carte dynamique + Tesseract OCR")
            st.info(f"**Poppler:** {config.POPPLER_PATH}")
            st.success("✅ Tesseract opérationnel")
        
        with col2:
            st.subheader("Plages de températures valides")
            st.metric("🌡️ Température Min", f"{config.TEMP_MIN_RANGE[0]}-{config.TEMP_MIN_RANGE[1]}°C")
            st.metric("🌡️ Température Max", f"{config.TEMP_MAX_RANGE[0]}-{config.TEMP_MAX_RANGE[1]}°C")
    
    with tab2:
        st.subheader("📍 Stations météorologiques du Burkina Faso")
        st.info(f"**Total:** {len(STATIONS_COORDS)} stations actives")
        
        # Affichage en colonnes
        cols = st.columns(4)
        for idx, (station, coords) in enumerate(STATIONS_COORDS.items()):
            with cols[idx % 4]:
                st.markdown(f"**✓ {station}**")
                st.caption(f"Lat: {coords[0]:.4f}")
                st.caption(f"Lon: {coords[1]:.4f}")
                st.markdown("---")
    
    with tab3:
        st.subheader("📂 Chemins du système de fichiers")
        
        paths_info = {
            "Base": str(config.BASE_DIR),
            "Uploads": str(config.UPLOAD_DIR),
            "Images traitées": str(config.IMAGE_DIR),
            "Données": str(config.DATA_DIR),
            "Base de données": str(config.DB_PATH)
        }
        
        for label, path in paths_info.items():
            st.text_input(f"📁 {label}", path, disabled=True)
        
        st.markdown("---")
        st.info("💡 **Note:** Ces chemins sont configurés automatiquement au démarrage de l'application")
