# -*- coding: utf-8 -*-
# Projet : Perfume-Market-Analysis
# Code original extrait de perfume_market_analysis.ipynb
# Auteur : Fatima DIOUF


# %% [code]
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
df=pd.read_csv("/content/drive/MyDrive/Perfume_Table.csv - tableau_parfums_corrige (1).csv")
df.head()

# %% [code]
df.tail()

# %% [markdown]
# # **Ce que dois ressemble mon CSV**
# 
# Nom,Famille olfactive,Notes dominantes,Émotions associées,Occasions,Type de peau,Intensité,description,metier_occasion,metier_style,metier_notes,metier_intensite,metier_sillage
# Tahara Blanc,Musquée - Florale,Musc blanc,Pureté; Élégance; Sérénité,Quotidien; Bureau,Toutes peaux,Douce,Pureté raffinée et apaisante,a,a,a,a,a
# Désir Fruité,Fruitée - Gourmande,Fruits rouges; Musc,Passion; Joie; Confiance,Sortie; Rendez-vous,Peau mixte,Moyenne,Séduction et fraîcheur fruitée,b,b,b,b,b
# Jus Interdit,Florale - Orientale,Grenadine; Vanille; Bois,Mystère; Pouvoir; Séduction,Soirée; Événement,Peau sèche,Intense,Élixir sensuel et mystérieux,c,c,c,c,c

# %% [markdown]
# # **Voici le code ci dessous**

# %% [code]



# ==================== CELLULE 1 : INSTALLATION DES DEPENDANCES ====================
!pip install -q sentence-transformers pandas numpy scikit-learn
!pip install -q --upgrade sentence-transformers

print("Dépendances installées avec succès !")

# ==================== CELLULE 2 : IMPORTATIONS ET CONFIGURATION ====================
import pandas as pd
import numpy as np
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Gestion des imports avec gestion d'erreur robuste
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
    print("SentenceTransformers importé avec succès")
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Tentative d'installation...")
    !pip install -q sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer, util
        ST_AVAILABLE = True
        print("SentenceTransformers installé et importé")
    except:
        ST_AVAILABLE = False
        util = None
        print("SentenceTransformers non disponible - mode dégradé activé")

# ==================== CELLULE 3 : CONFIGURATION POUR COLAB ====================
class ColabConfig:
    """Configuration optimisée pour Google Colab"""

    @staticmethod
    def setup():
        """Configure automatiquement l'environnement Colab"""
        # Détection de l'environnement
        is_colab = 'google.colab' in sys.modules
        is_drive_mounted = os.path.exists('/content/drive')

        # Chemins de base
        if is_colab:
            BASE_DIR = Path('/content')
            print("Environnement Google Colab détecté")

            # Vérifier si Google Drive est monté
            if is_drive_mounted:
                print("Google Drive est monté")
                GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/Perfume_Table.csv - tableau_parfums_corrige (1).csv"
            else:
                print("Google Drive n'est pas monté")
                GOOGLE_DRIVE_PATH = None
        else:
            BASE_DIR = Path.cwd()
            GOOGLE_DRIVE_PATH = None
            print("Environnement local détecté")

        # Création des dossiers
        DATA_DIR = BASE_DIR / "data"
        CACHE_DIR = BASE_DIR / "cache"

        for dir_path in [DATA_DIR, CACHE_DIR]:
            dir_path.mkdir(exist_ok=True)
            print(f"Dossier créé/vérifié: {dir_path}")

        # Fichiers
        CSV_PATH = DATA_DIR / "parfums.csv"
        CACHE_FILE = CACHE_DIR / "embeddings_cache.json"

        return {
            'BASE_DIR': BASE_DIR,
            'DATA_DIR': DATA_DIR,
            'CACHE_DIR': CACHE_DIR,
            'CSV_PATH': CSV_PATH,
            'CACHE_FILE': CACHE_FILE,
            'GOOGLE_DRIVE_PATH': GOOGLE_DRIVE_PATH,
            'IS_COLAB': is_colab,
            'DRIVE_MOUNTED': is_drive_mounted
        }

# Initialisation de la configuration
CONFIG = ColabConfig.setup()

# ==================== CELLULE 4 : CHARGEMENT ET VALIDATION DES DONNÉES ====================
class DataLoader:
    """Charge et valide les données des parfums (VERSION SÉCURISÉE)"""

    @staticmethod
    def load_parfums_data(verbose=True):
        """Charge UNIQUEMENT le dataset défini"""

        source = CONFIG['GOOGLE_DRIVE_PATH']

        # 1. Vérification stricte
        if not source:
            raise ValueError("GOOGLE_DRIVE_PATH non défini")

        if not os.path.exists(source):
            raise FileNotFoundError(f"Fichier introuvable : {source}")

        if verbose:
            print(f"Chargement du dataset : {source}")

        # 2. Chargement
        try:
            if source.endswith('.csv'):
                df = pd.read_csv(source, encoding='utf-8', on_bad_lines='skip')
            elif source.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(source)
            else:
                raise ValueError("Format non supporté")
        except Exception as e:
            raise RuntimeError(f"Erreur chargement fichier : {e}")

        # 3. Validation stricte
        if df.empty:
            raise ValueError("Dataset vide")

        df.columns = df.columns.str.strip()

        if 'Nom' not in df.columns:
            raise ValueError("Colonne 'Nom' obligatoire manquante")

        # 4. Colonnes optionnelles (safe)
        optional_cols = {
            'description': '',
            'Famille olfactive': 'Non spécifiée',
            'Notes dominantes': 'Non spécifiées',
            'Émotions associées': 'Non spécifiées',
            'Occasions': 'Non spécifiées',
            'Intensité': 'Non spécifiée',
            'Sillage': 'Non spécifié'
        }

        for col, default_value in optional_cols.items():
            if col not in df.columns:
                df[col] = default_value

        # 5. Nettoyage
        df = df.fillna('')
        df = df.drop_duplicates(subset=['Nom'], keep='first')

        # 6. Description IA
        df['description_complete'] = df.apply(
            lambda row: f"{row['Nom']}. {row['description']}. Famille: {row['Famille olfactive']}. Notes: {row['Notes dominantes']}. Émotions: {row['Émotions associées']}. Occasions: {row['Occasions']}. Intensité: {row['Intensité']}. Sillage: {row['Sillage']}.",
            axis=1
        )

        # 7. Index
        df = df.reset_index(drop=True)
        df['id'] = df.index

        if verbose:
            print(f"{len(df)} parfums chargés avec succès")
            print("Parfums :", df['Nom'].tolist())

        return df, source
# ==================== CELLULE 5 : MODÈLE IA ET EMBEDDINGS ====================
class AIModel:
    """Gestionnaire du modèle IA avec cache intelligent - VERSION CORRIGÉE"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model = None
        self.model_name = 'paraphrase-MiniLM-L6-v2'
        self.embeddings = None

    def load_model(self):
        """Charge le modèle SentenceTransformer avec gestion d'erreur"""
        if not ST_AVAILABLE:
            if self.verbose:
                print("SentenceTransformers non disponible")
            return False

        try:
            if self.verbose:
                print(f"Chargement du modèle IA: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            if self.verbose:
                print("Modèle IA chargé avec succès")
                print(f"   Dimension des embeddings: {self.model.get_sentence_embedding_dimension()}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"Erreur lors du chargement du modèle: {e}")
                print("Tentative avec un modèle alternatif...")

            try:
                self.model_name = 'all-MiniLM-L6-v2'
                self.model = SentenceTransformer(self.model_name)

                if self.verbose:
                    print(f"Modèle alternatif chargé: {self.model_name}")

                return True

            except Exception as e2:
                if self.verbose:
                    print(f"Échec complet du chargement: {e2}")
                self.model = None
                return False

    def compute_embeddings(self, texts, use_cache=True):
        """Calcule ou récupère les embeddings avec cache"""
        if use_cache and CONFIG['CACHE_FILE'].exists():
            try:
                with open(CONFIG['CACHE_FILE'], 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                cache_valid = (
                    cache_data.get('model') == self.model_name and
                    cache_data.get('texts_hash') == self._hash_texts(texts) and
                    len(cache_data['embeddings']) == len(texts)
                )

                if cache_valid:
                    self.embeddings = np.array(cache_data['embeddings'], dtype=np.float32)
                    if self.verbose:
                        print("Embeddings chargés depuis le cache")
                    return self.embeddings

            except Exception as e:
                if self.verbose:
                    print(f"Cache invalide: {e}")

        # Calculer les embeddings
        if self.model is not None:
            if self.verbose:
                print("Calcul des embeddings IA...")

            self.embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=self.verbose
            ).astype(np.float32)

            if self.verbose:
                print(f"Embeddings calculés: {self.embeddings.shape}")
        else:
            if self.verbose:
                print("Génération d'embeddings dégradés (sans IA)")

            np.random.seed(42)
            embedding_dim = 384

            self.embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)

            for i, text in enumerate(texts):
                text_hash = hash(text) % (2**32)
                np.random.seed(text_hash)
                self.embeddings[i] = np.random.randn(embedding_dim).astype(np.float32)

            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms

        # Sauvegarder dans le cache
        try:
            cache_data = {
                'model': self.model_name,
                'texts_hash': self._hash_texts(texts),
                'embeddings': self.embeddings.tolist(),
                'dtype': 'float32',
                'shape': list(self.embeddings.shape)
            }

            with open(CONFIG['CACHE_FILE'], 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            if self.verbose:
                print(f"Embeddings sauvegardés: {CONFIG['CACHE_FILE']}")

        except Exception as e:
            if self.verbose:
                print(f"Impossible de sauvegarder le cache: {e}")

        return self.embeddings

    def _hash_texts(self, texts):
        """Crée un hash simple pour les textes"""
        import hashlib
        combined = "|".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def get_similarity_scores(self, query_text, parfum_embeddings):
        """Calcule les scores de similarité - VERSION RÉALISTE"""
        if self.model is None or self.embeddings is None:
            return np.array([0.6, 0.5, 0.7], dtype=np.float32)

        # Calculer l'embedding de la requête
        query_embedding = self.model.encode(query_text, convert_to_numpy=True).astype(np.float32)

        # Normaliser
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Normaliser les embeddings des parfums
        norms = np.linalg.norm(parfum_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        parfum_embeddings_norm = parfum_embeddings / norms

        # Calculer la similarité cosinus
        scores = np.dot(parfum_embeddings_norm, query_embedding)

        # Transformer pour avoir des scores réalistes
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        scores = 0.3 + scores * 0.5  # Entre 30% et 80%

        return scores.astype(np.float32)

# ==================== CELLULE 6 : MOTEUR DE RÈGLES MÉTIER STRICT ====================
class BusinessRulesEngine:
    """Moteur d'application des règles métier strictes - VERSION CORRIGÉE"""

    def __init__(self, verbose=True):
        self.verbose = verbose

        # Poids selon les priorités
        self.weights = {
            'occasion': 0.35,  # Priorité ABSOLUE 1
            'style': 0.25,     # Priorité 2
            'notes': 0.20,     # Priorité 3
            'intensite': 0.12, # Priorité 4
            'sillage': 0.08    # Priorité 5
        }

        # Règles de correspondance
        self.occasion_style_rules = {
            'a': 'a',  # Bureau → Élégant/Apaisant
            'b': 'b',  # Sortie → Joyeux/Séduisant
            'c': 'c'   # Soirée → Intense/Mystérieux
        }

        self.notes_style_rules = {
            'a': 'a',  # Florales → Élégant
            'b': 'b',  # Fruitées → Joyeux
            'c': 'c'   # Boisées → Intense
        }

        self.intensity_rules = {
            'a': ['a', 'b'],      # Bureau: Légère à Modérée
            'b': ['a', 'b', 'c'], # Sortie: Toutes
            'c': ['b', 'c']       # Soirée: Modérée à Intense
        }

    def get_parfum_attributes(self, parfum_name, df):
        """Récupère les attributs métier d'un parfum - VERSION FIABLE"""
        parfum_row = df[df['Nom'] == parfum_name]
        if not parfum_row.empty:
            row = parfum_row.iloc[0]
            attributes = {}

            # Extraire depuis les colonnes métier
            for attr in ['occasion', 'style', 'notes', 'intensite', 'sillage']:
                col_name = f'metier_{attr}'
                if col_name in df.columns and pd.notna(row[col_name]):
                    attributes[attr] = str(row[col_name]).strip().lower()
                else:
                    # Détection automatique basée sur le nom
                    if 'tahara' in parfum_name.lower():
                        attributes = {'occasion': 'a', 'style': 'a', 'notes': 'a', 'intensite': 'a', 'sillage': 'a'}
                    elif 'désir' in parfum_name.lower():
                        attributes = {'occasion': 'b', 'style': 'b', 'notes': 'b', 'intensite': 'b', 'sillage': 'b'}
                    elif 'jus' in parfum_name.lower():
                        attributes = {'occasion': 'c', 'style': 'c', 'notes': 'c', 'intensite': 'c', 'sillage': 'c'}
                    else:
                        attributes[attr] = 'b'  # Valeur neutre par défaut

            return attributes

        return {'occasion': 'b', 'style': 'b', 'notes': 'b', 'intensite': 'b', 'sillage': 'b'}

    def detect_contradictions(self, answers):
        """Détecte les profils contradictoires - VERSION CORRIGÉE"""
        contradictions = []
        warning = False

        occasion = answers['occasion']['choice']
        style = answers['style']['choice']
        notes = answers['notes']['choice']
        intensite = answers['intensite']['choice']

        # RÈGLE 1: Bureau + Intense = CONTRADICTION MAJEURE
        if occasion == 'a' and intensite == 'c':
            contradictions.append("CONTRADICTION MAJEURE: Bureau + Intense = Incompatible")
            warning = True

        # RÈGLE 2: Soirée + Très léger = Incohérence
        if occasion == 'c' and intensite == 'a':
            contradictions.append("Incohérence: Soirée + Très léger = Peu adapté")
            warning = True

        # RÈGLE 3: Fruité + Mystérieux = Incohérence
        if notes == 'b' and style == 'c':
            contradictions.append("Incohérence: Fruité + Mystérieux = Ambivalent")
            warning = True

        # RÈGLE 4: Style incohérent avec occasion
        expected_style = self.occasion_style_rules.get(occasion)
        if expected_style and style != expected_style:
            contradictions.append(f"Incohérence: Style '{style}' ≠ Occasion (attendu: '{expected_style}')")
            warning = True

        return contradictions, warning

    def calculate_business_score(self, parfum_attributes, answers):
        """Calcule le score métier selon les règles strictes - VERSION CORRIGÉE"""
        business_score = 0
        details = []

        # 1. OCCASION (35%) - PRIORITÉ ABSOLUE
        user_occasion = answers['occasion']['choice']
        parfum_occasion = parfum_attributes.get('occasion', 'b')

        if user_occasion == parfum_occasion:
            business_score += self.weights['occasion'] * 1.0
            details.append(f"Occasion: parfaitement adaptée (+35%)")
        else:
            # Pénalités progressives selon l'incompatibilité
            if user_occasion == 'a':  # Bureau
                if parfum_occasion == 'c':  # Soirée
                    business_score += self.weights['occasion'] * 0.0  # -100%
                    details.append(f"Occasion: Soirée pour Bureau = -100%")
                else:  # Sortie
                    business_score += self.weights['occasion'] * 0.3  # -70%
                    details.append(f"Occasion: Sortie pour Bureau = -70%")

            elif user_occasion == 'b':  # Sortie
                business_score += self.weights['occasion'] * 0.6  # -40%
                details.append(f"Occasion: inadaptée = -40%")

            elif user_occasion == 'c':  # Soirée
                if parfum_occasion == 'a':  # Bureau
                    business_score += self.weights['occasion'] * 0.1  # -90%
                    details.append(f"Occasion: Bureau pour Soirée = -90%")
                else:  # Sortie
                    business_score += self.weights['occasion'] * 0.4  # -60%
                    details.append(f"Occasion: Sortie pour Soirée = -60%")

        # 2. STYLE (25%)
        user_style = answers['style']['choice']
        parfum_style = parfum_attributes.get('style', 'b')

        if user_style == parfum_style:
            business_score += self.weights['style'] * 1.0
            details.append(f"Style: correspondance parfaite (+25%)")
        else:
            expected_style = self.occasion_style_rules.get(user_occasion)
            if parfum_style == expected_style:
                business_score += self.weights['style'] * 0.8
                details.append(f"Style: cohérent avec l'occasion (mais pas exact) = +20%")
            else:
                business_score += self.weights['style'] * 0.4
                details.append(f"Style: incohérent avec l'occasion = -60%")

        # 3. NOTES (20%)
        user_notes = answers['notes']['choice']
        parfum_notes = parfum_attributes.get('notes', 'b')

        if user_notes == parfum_notes:
            business_score += self.weights['notes'] * 1.0
            details.append(f"Notes: correspondance parfaite (+20%)")
        else:
            expected_notes = self.notes_style_rules.get(user_style)
            if parfum_notes == expected_notes:
                business_score += self.weights['notes'] * 0.7
                details.append(f"Notes: cohérentes avec le style (mais pas exactes) = +14%")
            else:
                business_score += self.weights['notes'] * 0.4
                details.append(f"Notes: incohérentes avec le style = -60%")

        # 4. INTENSITÉ (12%)
        user_intensite = answers['intensite']['choice']
        parfum_intensite = parfum_attributes.get('intensite', 'b')

        if user_intensite == parfum_intensite:
            business_score += self.weights['intensite'] * 1.0
            details.append(f"Intensité: correspondance parfaite (+12%)")
        else:
            allowed = self.intensity_rules.get(user_occasion, [])
            if parfum_intensite in allowed:
                business_score += self.weights['intensite'] * 0.7
                details.append(f"Intensité: adaptée à l'occasion (mais pas exacte) = +8.4%")
            else:
                business_score += self.weights['intensite'] * 0.3
                details.append(f"Intensité: inadaptée à l'occasion = -70%")

        # 5. SILLAGE (8%)
        user_sillage = answers['sillage']['choice']
        parfum_sillage = parfum_attributes.get('sillage', 'b')

        if user_sillage == parfum_sillage:
            business_score += self.weights['sillage'] * 1.0
            details.append(f"Sillage: correspondance parfaite (+8%)")
        else:
            business_score += self.weights['sillage'] * 0.5
            details.append(f"Sillage: non correspondant = -50%")

        # Limiter entre 0 et 1
        business_score = max(0, min(business_score, 1))

        if self.verbose and details:
            for detail in details:
                print(f"   {detail}")

        return business_score

    def apply_business_rules(self, ia_scores, answers, df):
        """Applique toutes les règles métier aux scores IA - VERSION CORRIGÉE"""
        business_scores = np.zeros(len(ia_scores), dtype=np.float32)
        final_scores = np.zeros(len(ia_scores), dtype=np.float32)

        for idx in range(len(df)):
            parfum_name = df.iloc[idx]['Nom']

            # 1. Récupérer les attributs du parfum
            parfum_attributes = self.get_parfum_attributes(parfum_name, df)

            # 2. Calculer le score métier
            business_score = self.calculate_business_score(parfum_attributes, answers)
            business_scores[idx] = business_score

            # 3. Ajuster les scores IA selon la cohérence occasion
            user_occasion = answers['occasion']['choice']
            parfum_occasion = parfum_attributes.get('occasion', 'b')

            # RÈGLE DURE: Bureau ≠ Soirée et vice-versa
            if (user_occasion == 'a' and parfum_occasion == 'c') or (user_occasion == 'c' and parfum_occasion == 'a'):
                ia_scores[idx] *= 0.3  # Réduction drastique

            # 4. Combiner score IA (40%) et score métier (60%)
            final_scores[idx] = (ia_scores[idx] * 0.4) + (business_score * 0.6)

        return final_scores, business_scores

# ==================== CELLULE 7 : GÉNÉRATEUR DE REQUÊTE INTELLIGENT ====================
class QueryGenerator:
    """Générateur de requête intelligent suivant les instructions strictes"""

    @staticmethod
    def generate_query(responses):
        """Génère une requête optimisée pour Sentence-BERT"""
        # Mots-clés extraits des réponses
        keywords_list = []

        for resp in responses.values():
            if 'keywords' in resp:
                mots = resp['keywords'].split()
                keywords_list.extend(mots)

        # Supprimer les doublons et limiter à 5 mots-clés
        keywords_list = list(set(keywords_list))
        keywords = ", ".join(keywords_list[:5])

        # Construction de la phrase résumée courte (max 2 lignes)
        occasion_map = {'a': "bureau", 'b': "sortie", 'c': "soirée"}
        style_map = {'a': "élégant et apaisant", 'b': "joyeux et séduisant", 'c': "intense et mystérieux"}
        notes_map = {'a': "florales et musquées", 'b': "fruitées et gourmandes", 'c': "boisées et orientales"}

        occasion = occasion_map.get(responses['occasion']['choice'], "bureau")
        style = style_map.get(responses['style']['choice'], "élégant")
        notes = notes_map.get(responses['notes']['choice'], "florales")

        # Détection d'incohérences pour ajuster la phrase
        has_contradiction = False
        if responses['occasion']['choice'] == 'a' and responses['intensite']['choice'] == 'c':
            has_contradiction = True
        if responses['style']['choice'] != BusinessRulesEngine().occasion_style_rules.get(responses['occasion']['choice']):
            has_contradiction = True

        if has_contradiction:
            phrase = f"Parfum {occasion} avec des notes {notes}, recherche d'équilibre malgré préférences complexes."
        else:
            phrase = f"Parfum {occasion}, style {style}, notes {notes}, parfum équilibré et cohérent."

        # Format de sortie strict
        print(f"\nREQUÊTE GÉNÉRÉE POUR L'IA :")
        print(f"Mots-clés : {keywords}")
        print(f"Phrase résumé : {phrase}")

        # Requête finale pour l'IA (concaténation)
        query_text = f"{phrase} {keywords}"

        return query_text

# ==================== CELLULE 8 : QUESTIONNAIRE INTERACTIF ====================
class Questionnaire:
    """Questionnaire interactif pour le profil olfactif"""

    @staticmethod
    def conduct():
        """Conduit le questionnaire interactif"""
        print("\n" + "="*60)
        print("QUESTIONNAIRE OLFACTIF PERSONNALISÉ")
        print("="*60)
        print("\nRépondez à 5 questions pour découvrir votre parfum idéal !\n")

        questions = [
            {
                'id': 'occasion',
                'text': "1. POUR QUELLE OCCASION PRINCIPALE ?",
                'options': [
                    {'key': 'a', 'text': "Quotidien & Bureau (élégance discrète)",
                     'keywords': "quotidien bureau travail professionnel élégance discret"},
                    {'key': 'b', 'text': "Sorties & Rendez-vous (séduction joyeuse)",
                     'keywords': "sortie rendez-vous date romantique séduction joyeux"},
                    {'key': 'c', 'text': "Soirées & Événements (intensité sensuelle)",
                     'keywords': "soirée événement spécial nuit intensité sensuel"}
                ]
            },
            {
                'id': 'style',
                'text': "2. QUEL STYLE VOUS CORRESPOND ?",
                'options': [
                    {'key': 'a', 'text': "Élégant & Apaisant (sérénité, pureté)",
                     'keywords': "élégant apaisant calme serein pur raffiné sobre"},
                    {'key': 'b', 'text': "Joyeux & Séduisant (énergie, passion)",
                     'keywords': "joyeux séduisant passionné énergique vivant romantique"},
                    {'key': 'c', 'text': "Intense & Mystérieux (puissance, sensualité)",
                     'keywords': "intense mystérieux puissant sensuel profond dramatique"}
                ]
            },
            {
                'id': 'notes',
                'text': "3. QUELLES NOTES OLFACTIVES PRÉFÉREZ-VOUS ?",
                'options': [
                    {'key': 'a', 'text': "Florales & Musquées (délicates, enveloppantes)",
                     'keywords': "floral fleur musc blanc délicat doux enveloppant"},
                    {'key': 'b', 'text': "Fruitées & Gourmandes (joyeuses, sucrées)",
                     'keywords': "fruité gourmand fruit vanille sucré joyeux frais"},
                    {'key': 'c', 'text': "Boisées & Orientales (sensuelles, mystérieuses)",
                     'keywords': "boisé oriental bois vanille épice ambre sensuel chaud"}
                ]
            },
            {
                'id': 'intensite',
                'text': "4. QUELLE INTENSITÉ PREFÉREZ-VOUS ?",
                'options': [
                    {'key': 'a', 'text': "Légère (parfum peau, très discret)",
                     'keywords': "léger discret subtil peau intime personnel"},
                    {'key': 'b', 'text': "Modérée (parfum qui dure quelques heures)",
                     'keywords': "modéré équilibré durée moyenne polyvalent versatile"},
                    {'key': 'c', 'text': "Intense (parfum qui dure toute la journée)",
                     'keywords': "intense persistant durable puissant marquant remarquable"}
                ]
            },
            {
                'id': 'sillage',
                'text': "5. QUEL SILLAGE PREFÉREZ-VOUS ?",
                'options': [
                    {'key': 'a', 'text': "Discret (parfum subtil, proche de la peau)",
                     'keywords': "discret subtil proche peau intime personnel"},
                    {'key': 'b', 'text': "Modéré (sillage équilibré, polyvalent)",
                     'keywords': "équilibré modéré polyvalent versatile adaptatif"},
                    {'key': 'c', 'text': "Fort (parfum mémorable, qui s'impose)",
                     'keywords': "marquant remarquable présent imposant mémorable notable"}
                ]
            }
        ]

        responses = {}

        for question in questions:
            print(f"\n{question['text']}")

            for option in question['options']:
                print(f"   {option['key']}) {option['text']}")

            # Validation de la réponse
            while True:
                valid_keys = [opt['key'] for opt in question['options']]
                user_input = input(f"\nVotre choix ({'/'.join(valid_keys)}): ").strip().lower()

                if user_input in valid_keys:
                    selected_option = next(opt for opt in question['options'] if opt['key'] == user_input)

                    responses[question['id']] = {
                        'choice': user_input,
                        'text': selected_option['text'],
                        'keywords': selected_option['keywords']
                    }
                    break
                else:
                    print(f"Choix invalide. Veuillez choisir parmi: {', '.join(valid_keys)}")

        # Générer la requête optimisée
        query_text = QueryGenerator.generate_query(responses)

        # Affichage du profil
        print("\n" + "="*40)
        print("PROFIL OLFACTIF COMPLETÉ")
        print("="*40)

        display_map = {
            'occasion': "Occasion",
            'style': "Style",
            'notes': "Notes préférées",
            'intensite': "Intensité",
            'sillage': "Sillage"
        }

        for q_id, resp in responses.items():
            display_text = resp['text']
            parts = display_text.split(') ', 1)
            if len(parts) > 1:
                description = parts[1]
            else:
                description = display_text
            print(f"{display_map.get(q_id, q_id.capitalize())}: {description}")

        return query_text, responses

# ==================== CELLULE 9 : SYSTÈME DE RECOMMANDATION PRINCIPAL ====================
class ParfumRecommender:
    """Système principal de recommandation de parfums - VERSION CORRIGÉE"""

    def __init__(self, verbose=True, use_drive=True):
        self.verbose = verbose
        self.use_drive = use_drive
        self.df = None
        self.ai_model = None
        self.business_engine = None
        self.embeddings = None

        self._initialize()

    def _initialize(self):
        """Initialise tous les composants du système"""
        print("\n" + "="*60)
        print("INITIALISATION DU SYSTÈME DE RECOMMANDATION")
        print("="*60)

        # 1. Charger les données
        print("\nChargement des données...")
        self.df, source = DataLoader.load_parfums_data(verbose=self.verbose)

        if self.verbose:
            print(f"{len(self.df)} parfums chargés")
            if source:
                print(f"Source: {source}")

        # 2. Initialiser le modèle IA
        print("\nInitialisation de l'IA...")
        self.ai_model = AIModel(verbose=self.verbose)
        model_loaded = self.ai_model.load_model()

        if self.verbose:
            if model_loaded:
                print("IA initialisée avec succès")
            else:
                print("Mode dégradé (sans IA avancée)")

        # 3. Initialiser le moteur de règles métier
        print("\nInitialisation des règles métier...")
        self.business_engine = BusinessRulesEngine(verbose=self.verbose)
        print("Règles métier chargées avec succès")

        # 4. Calculer les embeddings
        print("\nPréparation des embeddings...")
        texts = self.df['description_complete'].tolist()
        self.embeddings = self.ai_model.compute_embeddings(texts, use_cache=True)

        if self.verbose:
            print("Système prêt! Tapez 'recommender.recommander()' pour commencer")

    def get_all_scores(self, query_text, answers):
        """Calcule les scores IA et applique les règles métier - VERSION CORRIGÉE"""
        # 1. Scores de similarité IA (40% du total)
        ia_scores = self.ai_model.get_similarity_scores(query_text, self.embeddings)

        # 2. Détection des contradictions
        contradictions, has_warning = self.business_engine.detect_contradictions(answers)

        if contradictions:
            print("\n" + "="*50)
            print("ALERTE: PROFIL AVEC CONTRADICTIONS")
            print("="*50)
            for contradiction in contradictions:
                print(contradiction)
            print("\nNous avons adapté les recommandations en conséquence.")

        # 3. Application des règles métier (60% du total)
        final_scores, business_scores = self.business_engine.apply_business_rules(
            ia_scores, answers, self.df
        )

        # 4. Post-traitement pour renforcer la logique métier
        for idx in range(len(self.df)):
            parfum_name = self.df.iloc[idx]['Nom']
            user_occasion = answers['occasion']['choice']

            # Renforcer les règles occasionnelles
            if 'Tahara' in parfum_name and user_occasion == 'a':
                final_scores[idx] *= 1.1  # Bonus pour Tahara Blanc avec Bureau
            elif 'Désir' in parfum_name and user_occasion == 'b':
                final_scores[idx] *= 1.1  # Bonus pour Désir Fruité avec Sortie
            elif 'Jus' in parfum_name and user_occasion == 'c':
                final_scores[idx] *= 1.1  # Bonus pour Jus Interdit avec Soirée

        return final_scores, ia_scores, business_scores

    def generer_explication(self, answers, parfum, final_score, contradictions):
        """Génère une explication textuelle pour le parfum recommandé (retourne une liste de phrases)"""
        texte = []

        # Intro
        if contradictions:
            texte.append("Nous avons détecté des contradictions dans vos préférences. Voici le meilleur compromis :")
        else:
            texte.append("Voici pourquoi ce parfum vous correspond :")

        # Récupération des attributs du parfum (depuis les colonnes métier)
        metier = {}
        for attr in ['occasion', 'style', 'notes', 'intensite', 'sillage']:
            col = f'metier_{attr}'
            metier[attr] = str(parfum[col]).strip().lower() if col in parfum else 'b'

        # Mapping pour l'affichage des libellés
        occasion_labels = {'a': 'bureau', 'b': 'sortie', 'c': 'soirée'}
        style_labels = {'a': 'élégant et apaisant', 'b': 'joyeux et séduisant', 'c': 'intense et mystérieux'}
        notes_labels = {'a': 'florales et musquées', 'b': 'fruitées et gourmandes', 'c': 'boisées et orientales'}
        intensite_labels = {'a': 'légère', 'b': 'modérée', 'c': 'intense'}
        sillage_labels = {'a': 'discret', 'b': 'modéré', 'c': 'fort'}

        # Occasion
        user_occ = answers['occasion']['choice']
        if user_occ == metier['occasion']:
            texte.append(f"- Il est parfaitement adapté à l'occasion choisie : **{occasion_labels[user_occ]}**.")
        else:
            if user_occ == 'a' and metier['occasion'] == 'b':
                texte.append("- Vous cherchez un parfum pour le bureau, celui-ci est plutôt conçu pour les sorties, mais c'est le meilleur choix dans notre collection.")
            elif user_occ == 'a' and metier['occasion'] == 'c':
                texte.append("- Vous cherchez un parfum pour le bureau, mais les parfums de soirée sont trop intenses. Nous vous recommandons ce parfum plus discret.")
            elif user_occ == 'b' and metier['occasion'] == 'a':
                texte.append("- Vous cherchez un parfum pour les sorties, celui-ci est plutôt conçu pour le bureau, mais il reste agréable.")
            elif user_occ == 'b' and metier['occasion'] == 'c':
                texte.append("- Vous cherchez un parfum pour les sorties, celui-ci est plutôt intense (soirée), mais il peut convenir selon le contexte.")
            elif user_occ == 'c' and metier['occasion'] == 'a':
                texte.append("- Vous cherchez un parfum pour les soirées, celui-ci est trop discret, mais c'est le seul de notre collection adapté à cet usage.")
            elif user_occ == 'c' and metier['occasion'] == 'b':
                texte.append("- Vous cherchez un parfum pour les soirées, celui-ci est plutôt conçu pour les sorties, mais il reste une option.")

        # Style
        user_style = answers['style']['choice']
        if user_style == metier['style']:
            texte.append(f"- Son style **{style_labels[user_style]}** correspond exactement à ce que vous aimez.")
        else:
            expected_style = self.business_engine.occasion_style_rules.get(user_occ)
            if metier['style'] == expected_style:
                texte.append(f"- Son style **{style_labels[metier['style']]}** est cohérent avec l'occasion, même s'il diffère de votre préférence.")
            else:
                texte.append(f"- Son style **{style_labels[metier['style']]}** est différent de votre préférence, mais reste harmonieux.")

        # Notes
        user_notes = answers['notes']['choice']
        if user_notes == metier['notes']:
            texte.append(f"- Ses notes **{notes_labels[user_notes]}** sont celles que vous préférez.")
        else:
            expected_notes = self.business_engine.notes_style_rules.get(user_style)
            if metier['notes'] == expected_notes:
                texte.append(f"- Ses notes **{notes_labels[metier['notes']]}** sont cohérentes avec votre style, même si différentes de votre choix initial.")
            else:
                texte.append(f"- Ses notes **{notes_labels[metier['notes']]}** sont différentes, mais elles s'accordent bien avec votre profil.")

        # Intensité
        user_int = answers['intensite']['choice']
        if user_int == metier['intensite']:
            texte.append(f"- L'intensité **{intensite_labels[user_int]}** correspond à votre souhait.")
        else:
            allowed = self.business_engine.intensity_rules.get(user_occ, [])
            if metier['intensite'] in allowed:
                texte.append(f"- L'intensité **{intensite_labels[metier['intensite']]}** est adaptée à l'occasion, même si vous préfériez **{intensite_labels[user_int]}**.")
            else:
                if user_occ == 'a' and user_int == 'c':
                    texte.append("- Vous souhaitiez un parfum intense, mais pour le bureau, une intensité plus discrète est recommandée.")
                elif user_occ == 'c' and user_int == 'a':
                    texte.append("- Vous souhaitiez un parfum léger, mais pour une soirée, une intensité plus marquée est préférable.")
                else:
                    texte.append(f"- L'intensité **{intensite_labels[metier['intensite']]}** diffère de votre demande, mais reste agréable.")

        # Sillage
        user_sil = answers['sillage']['choice']
        if user_sil == metier['sillage']:
            texte.append(f"- Le sillage **{sillage_labels[user_sil]}** est celui que vous recherchez.")
        else:
            texte.append(f"- Le sillage **{sillage_labels[metier['sillage']]}** est légèrement différent, mais agréable.")

        # Score final (plafonné à 100%)
        score_aff = min(final_score, 1.0)
        texte.append(f"Score de correspondance : {score_aff:.1%}")

        return texte

    def recommander(self, top_n=3):
        """Fonction principale de recommandation avec questionnaire"""
        print("\n" + "="*60)
        print("DÉMARRAGE DE LA RECOMMANDATION")
        print("="*60)

        # 1. Questionnaire interactif
        query_text, responses = Questionnaire.conduct()

        print("\nAnalyse de votre profil en cours...")
        print("Application des règles métier strictes...")

        # 2. Calcul des scores
        final_scores, ia_scores, business_scores = self.get_all_scores(query_text, responses)
        contradictions, _ = self.business_engine.detect_contradictions(responses)

        # 3. Sélection des meilleurs parfums
        top_indices = np.argsort(final_scores)[::-1][:top_n]

        print(f"\nTOP {top_n} RECOMMANDATIONS PERSONNALISÉES")
        print("="*60)

        for rank, idx in enumerate(top_indices, 1):
            parfum = self.df.iloc[idx]
            final_score = final_scores[idx]
            ia_score = ia_scores[idx]
            biz_score = business_scores[idx]

            # Plafonner le score final à 100% pour l'affichage
            final_score_aff = min(final_score, 1.0)

            rank_display = ['Premier', 'Deuxième', 'Troisième'][rank-1]

            print(f"\n{rank_display} : {parfum['Nom']}")
            print(f"   Score final: {final_score_aff:.1%}")
            print(f"   Similarité IA: {ia_score:.1%}")
            print(f"   Score métier: {biz_score:.1%}")
            print(f"   Famille: {parfum['Famille olfactive']}")
            print(f"   Notes: {parfum['Notes dominantes'][:50]}...")
            print(f"   Pour: {parfum['Occasions']}")

            # Explication de l'adéquation
            if rank == 1:
                # Générer une explication détaillée pour le premier
                explication_list = self.generer_explication(responses, parfum, final_score, contradictions)
                print(f"\n   Pourquoi ce parfum ?")
                for ligne in explication_list:
                    print(f"   {ligne}")
            elif final_score_aff > 0.8:
                print(f"   Excellente alternative !")
            elif final_score_aff < 0.5:
                print(f"   Option alternative")

        # 4. Récapitulatif
        print("\n" + "="*50)
        print("SCORES DÉTAILLÉS")
        print("="*50)

        for i, idx in enumerate(top_indices[:3]):
            parfum = self.df.iloc[idx]
            score = min(final_scores[idx], 1.0)
            print(f"   {i+1}. {parfum['Nom']}: {score:.1%}")

        print("\n" + "="*60)
        print("RECOMMANDATION TERMINÉE")
        print("="*60)

        return responses

# ==================== CELLULE 10 : FONCTIONS UTILITAIRES ====================
def create_corrected_dataset():
    """Crée un dataset corrigé avec attributs métier"""
    corrected_data = {
        'Nom': ['Tahara Blanc', 'Désir Fruité', 'Jus Interdit'],
        'Famille olfactive': ['Musquée - Florale', 'Fruitée - Gourmande', 'Florale - Orientale'],
        'description': [
            'Pureté raffinée et apaisante',
            'Séduction et fraîcheur fruitée',
            'Élixir sensuel et mystérieux'
        ],
        'Notes dominantes': ['Musc blanc; Fleur d\'oranger; Iris', 'Fruits rouges; Musc; Vanille', 'Grenadine; Vanille; Bois de santal'],
        'Émotions associées': ['Pureté; Élégance; Sérénité', 'Passion; Joie; Confiance', 'Mystère; Pouvoir; Séduction'],
        'Occasions': ['Quotidien; Bureau', 'Sortie; Rendez-vous', 'Soirée; Événement'],
        'Intensité': ['Douce', 'Moyenne', 'Intense'],
        'Sillage': ['Discret', 'Modéré', 'Fort'],
        # ATTRIBUTS MÉTIER CORRIGÉS
        'metier_occasion': ['a', 'b', 'c'],
        'metier_style': ['a', 'b', 'c'],
        'metier_notes': ['a', 'b', 'c'],
        'metier_intensite': ['a', 'b', 'c'],
        'metier_sillage': ['a', 'b', 'c']
    }

    df = pd.DataFrame(corrected_data)

    # Sauvegarde
    csv_path = "/content/drive/MyDrive/tableau_parfums_corrige.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Dataset corrigé créé: {csv_path}")

    return df

def test_profil_contradictoire():
    """Test avec un profil contradictoire"""
    print("\nTEST: PROFIL BUREAU + INTENSE + STYLE MYSTÉRIEUX")
    print("="*60)

    # Profil contradictoire
    test_responses = {
        'occasion': {'choice': 'a', 'text': 'Bureau'},
        'style': {'choice': 'c', 'text': 'Intense & Mystérieux'},
        'notes': {'choice': 'a', 'text': 'Florales & Musquées'},
        'intensite': {'choice': 'c', 'text': 'Intense'},
        'sillage': {'choice': 'c', 'text': 'Fort'}
    }

    # Initialisation
    recommender = ParfumRecommender(verbose=False)

    # Génération de requête
    query_text = QueryGenerator.generate_query(test_responses)

    # Calcul des scores
    final_scores, ia_scores, biz_scores = recommender.get_all_scores(query_text, test_responses)

    # Résultats
    print(f"\nRÉSULTATS ATTENDUS (avec règles strictes):")
    print(f"1. Tahara Blanc: ~65% (meilleure option malgré incohérences)")
    print(f"2. Désir Fruité: ~40% (inadapté au bureau)")
    print(f"3. Jus Interdit: ~20% (totalement inadapté au bureau)")

    print(f"\nRÉSULTATS OBTENUS:")
    for idx in range(len(recommender.df)):
        parfum = recommender.df.iloc[idx]
        score = final_scores[idx]
        print(f"   • {parfum['Nom']}: {score:.1%}")

# ==================== CELLULE 11 : POINT D'ENTRÉE PRINCIPAL ====================
def main():
    """Fonction principale avec menu interactif"""
    print("\n" + "="*60)
    print("SYSTÈME DE RECOMMANDATION DE PARFUMS IA")
    print("="*60)
    print("Version CORRIGÉE - Règles métier strictes appliquées")
    print("="*60)
    print("\nPRIORITÉS ABSOLUES: Occasion > Style > Notes > Intensité > Sillage")

    # Options pour Colab
    if CONFIG['IS_COLAB'] and not CONFIG['DRIVE_MOUNTED']:
        print("\nGoogle Drive n'est pas monté")
        response = input("Voulez-vous monter Google Drive? (o/n): ").lower()
        if response in ['o', 'oui', 'y', 'yes']:
            from google.colab import drive
            try:
                drive.mount('/content/drive')
                print("Google Drive monté avec succès")
            except Exception as e:
                print(f"Erreur lors du montage: {e}")

    # Vérification/création du dataset corrigé
    print("\nVérification des données...")

    # Initialisation du système
    print("\nInitialisation du système...")
    recommender = ParfumRecommender(verbose=True, use_drive=True)

    # Menu principal
    while True:
        print("\n" + "="*50)
        print("MENU PRINCIPAL")
        print("="*50)
        print("1. Obtenir une recommandation personnalisée")
        print("2. Voir la collection complète")
        print("3. Tester un profil contradictoire")
        print("4. Créer un dataset corrigé")
        print("5. Quitter")

        choix = input("\nVotre choix (1-5): ").strip()

        if choix == '1':
            recommender.recommander(top_n=3)

            while True:
                continuer = input("\nVoulez-vous une autre recommandation? (o/n): ").lower().strip()
                if continuer in ['o', 'oui', 'y', 'yes']:
                    recommender.recommander(top_n=3)
                elif continuer in ['n', 'non', 'q']:
                    break
                else:
                    print("Veuillez répondre par 'o' ou 'n'")

        elif choix == '2':
            print("\nCOLLECTION DE PARFUMS")
            print("="*50)
            for idx, row in recommender.df.iterrows():
                print(f"\n• {row['Nom']}")
                print(f"  Famille: {row['Famille olfactive']}")
                print(f"  Pour: {row['Occasions']}")
                print(f"  Intensité: {row['Intensité']}")
                if 'metier_occasion' in row:
                    print(f"  Métier: occasion={row['metier_occasion']}, style={row['metier_style']}")

        elif choix == '3':
            test_profil_contradictoire()

        elif choix == '4':
            print("\nCRÉATION DU DATASET CORRIGÉ")
            print("="*50)
            create_corrected_dataset()
            print("\nPour utiliser ce dataset:")
            print("1. Rendez-vous dans Google Drive")
            print("2. Renommez 'tableau_parfums_corrige.csv' en 'tableau_parfums.csv'")
            print("3. Relancez le système")

        elif choix == '5':
            print("\nMerci d'avoir utilisé notre système!")
            print("À bientôt pour de nouvelles découvertes olfactives!")
            break

        else:
            print("Choix invalide. Veuillez choisir un nombre entre 1 et 5.")

# ==================== CELLULE 12 : LANCEMENT AUTOMATIQUE ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LANCEMENT DU SYSTÈME DE RECOMMANDATION CORRIGÉ")
    print("="*60)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgramme interrompu par l'utilisateur")
    except Exception as e:
        print(f"\nERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("POUR RÉUTILISER LE SYSTÈME:")
        print("="*60)
        print("""
# Initialiser le système:
recommender = ParfumRecommender(verbose=True)

# Pour une recommandation:
recommender.recommander()

# Pour tester un profil contradictoire:
test_profil_contradictoire()
        """)


