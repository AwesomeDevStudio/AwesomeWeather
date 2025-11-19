README.md – AwesomeWeather
🌦️ AwesomeWeather – ANAM Burkina Faso
AwesomeWeather est une plateforme professionnelle de visualisation, d’analyse et d’évaluation automatisée des bulletins météorologiques du Burkina Faso, développée pour l’Agence Nationale de la Météorologie (ANAM).

🚀 Fonctionnalités clés
Tableau de bord interactif : valeurs clés, graphiques en temps réel, carte du Burkina Faso avec performances de chaque station météo !

Analyse automatisée de bulletins : upload PDF ou ZIP, extraction OCR, extraction et comparaison des températures par station.

Carte thermique mensuelle : suivez l’évolution des erreurs de prévision (MAE) par station et par période de l’année.

Historique complet : recherchez, consultez ou téléchargez tous vos bulletins analysés, avec détails station par station.

Statistiques et visualisations avancées : analyse journalière, mensuelle, annuelle, distribution, performances par station.

Configuration flexible : personnalisation des stations, paramètres techniques, chemins de travail.

⚙️ Installation
Clonez le repo :

bash
git clone https://github.com/AwesomeDevStudio/AwesomeWeather.git
cd AwesomeWeather
Créez un environnement virtuel et installez les dépendances :

bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
Lancez l’application Streamlit :

bash
streamlit run app.py
📁 Exemples d’utilisation
Uploader vos bulletins PDF/ZIP : extraction automatique des températures de toutes les stations, visualisation directe.

Analyser les performances : comparez la précision des prévisions sur la carte du Burkina Faso, suivez l’évolution par mois/année.

Exporter les résultats au format .json, .csv via l’interface.

📖 Résumé technique
Le code source de AwesomeWeather est organisé en modules Python :

app.py : interface principale avec Streamlit – gère la navigation, les uploads, l’affichage du dashboard, des analyses et de l’historique.

src/core/ : pipelines d’extraction OCR, traitement d’image (PDF -> images, extraction zones, température).

src/utils/ : gestion de la base de données, fonctions utilitaires.

config.py : configuration du projet (chemins, stations météo, options techniques).

La plateforme utilise Streamlit pour l’interface, Plotly/Folium pour la visualisation avancée, et Tesseract OCR pour l’analyse automatique des bulletins météo.

🛠️ Dépendances principales
Python 3.10+

streamlit

pandas

numpy

opencv-python

plotly

folium

streamlit-folium

pytesseract

pillow

Note : L’extraction PDF nécessite Tesseract et Poppler installés sur le système (hors requirements.txt).

📬 Contact
Responsable : Jean-Baptiste, AwesomeDevStudio

Agence Nationale de la Météorologie (ANAM), Burkina Faso

Email : contact@anam.bf
