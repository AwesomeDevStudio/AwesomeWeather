import sqlite3
import json
from datetime import datetime
from typing import List, Dict  # <- Ajouter cette ligne
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

def init_database():
    """Initialise la base de données SQLite"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bulletins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            extraction_date TEXT,
            stations_count INTEGER,
            status TEXT,
            data_json TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS temperatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bulletin_id INTEGER,
            station TEXT NOT NULL,
            temp_min INTEGER,
            temp_max INTEGER,
            validated BOOLEAN,
            FOREIGN KEY (bulletin_id) REFERENCES bulletins(id)
        )
    ''')
    
    conn.commit()
    conn.close()


def save_bulletin(filename: str, data: Dict, status: str = "completed"):
    """Sauvegarde un bulletin dans la base (observations + prévisions)"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    # Gérer structure dict avec observations/prévisions
    if isinstance(data, dict) and 'observations' in data:
        observations = data.get('observations', [])
        previsions = data.get('previsions', [])
        total_count = len(observations) + len(previsions)
    else:
        # Ancien format (liste simple)
        observations = data if isinstance(data, list) else []
        previsions = []
        total_count = len(observations)
    
    cursor.execute('''
        INSERT INTO bulletins (filename, upload_date, extraction_date, stations_count, status, data_json)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        datetime.now().isoformat(),
        datetime.now().isoformat(),
        total_count,
        status,
        json.dumps(data, ensure_ascii=False)
    ))
    
    bulletin_id = cursor.lastrowid
    
    # Sauvegarder observations
    for entry in observations:
        cursor.execute('''
            INSERT INTO temperatures (bulletin_id, station, temp_min, temp_max, validated)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            bulletin_id,
            f"{entry['station']} (OBS)",
            entry['temp_min'],
            entry['temp_max'],
            entry.get('validated', True)
        ))
    
    # Sauvegarder prévisions
    for entry in previsions:
        cursor.execute('''
            INSERT INTO temperatures (bulletin_id, station, temp_min, temp_max, validated)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            bulletin_id,
            f"{entry['station']} (PREV)",
            entry['temp_min'],
            entry['temp_max'],
            entry.get('validated', True)
        ))
    
    conn.commit()
    conn.close()
    
    return bulletin_id


def get_all_bulletins():
    """Récupère tous les bulletins"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM bulletins ORDER BY upload_date DESC')
    results = cursor.fetchall()
    
    conn.close()
    return results

def save_bulletin(filename: str, data: Dict, status: str = "completed"):
    """Sauvegarde un bulletin dans la base (observations + prévisions)"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    observations = data.get('observations', [])
    previsions = data.get('previsions', [])
    total_count = len(observations) + len(previsions)
    
    cursor.execute('''
        INSERT INTO bulletins (filename, upload_date, extraction_date, stations_count, status, data_json)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        datetime.now().isoformat(),
        datetime.now().isoformat(),
        total_count,
        status,
        json.dumps(data, ensure_ascii=False)
    ))
    
    bulletin_id = cursor.lastrowid
    
    # Sauvegarder observations
    for entry in observations:
        cursor.execute('''
            INSERT INTO temperatures (bulletin_id, station, temp_min, temp_max, validated)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            bulletin_id,
            f"{entry['station']} (OBS)",
            entry['temp_min'],
            entry['temp_max'],
            entry.get('validated', True)
        ))
    
    # Sauvegarder prévisions
    for entry in previsions:
        cursor.execute('''
            INSERT INTO temperatures (bulletin_id, station, temp_min, temp_max, validated)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            bulletin_id,
            f"{entry['station']} (PREV)",
            entry['temp_min'],
            entry['temp_max'],
            entry.get('validated', True)
        ))
    
    conn.commit()
    conn.close()
    
    return bulletin_id
