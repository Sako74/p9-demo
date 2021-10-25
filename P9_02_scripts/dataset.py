# -*- coding: utf-8 -*-
import os
import pickle
import tempfile
from datetime import datetime, timedelta

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import pandas as pd

from tqdm.notebook import tqdm

from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_RATING_COL = "rating_click_nb"


def get_clicks(clicks_dir):
    """Renvoie un DataFrame contenant les articles cliqués par les utilisateurs."""
    
    # On ouvre les fichiers et on ajoute les données dans une liste
    clicks = []
    for i in tqdm(os.listdir(clicks_dir), leave=False):
        tmp = pd.read_csv(os.path.join(clicks_dir, i))
        clicks.append(tmp)

    # On concatène toutes les données
    clicks = pd.concat(clicks)
    
    # On met à jour le type des variables
    clicks = clicks.astype({
        "user_id": np.uint64,
        "session_id": np.uint64,
        "session_start": np.uint64,
        "session_size": np.uint16,
        "click_article_id": np.uint64,
        "click_timestamp": np.uint64,
        "click_environment": np.uint8,
        "click_deviceGroup": np.uint8,
        "click_os": np.uint8,
        "click_country": np.uint8,
        "click_region": np.uint8,
        "click_referrer_type": np.uint8
    })
    
    # On convertit les timestamps en datetime
    clicks["session_start"] = pd.to_datetime(clicks['session_start'], unit='ms')
    clicks["click_timestamp"] = pd.to_datetime(clicks['click_timestamp'], unit='ms')
    
    # On renomme les colonnes
    clicks = clicks.rename(columns={
        "session_start": "session_start_dt",
        "click_timestamp": "click_dt",
    })
    
    # On trie par ordre chronologique
    clicks = clicks.sort_values("click_dt")
    
    return clicks

def upload_clicks_in_datastore(clicks, datastore, show_progress=True):
    """"""
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # On regroupe les données par jour
        grp = clicks.groupby(pd.Grouper(key="click_dt", freq="D"))

        # On enregistre les dataframes
        for i, df in tqdm(grp, total=len(grp), leave=False):
            # On crée un dossier avec une arborescence de type date
            dir_name = os.path.join(tmp_dir_name, i.strftime("%Y/%m/%d"))
            os.makedirs(dir_name)

            # On enregistre les données. "coerce_timestamps" permet de conserver
            # le type datetime dans les métadonnées du fichier parquet.
            df.to_parquet(os.path.join(dir_name, "data.parquet"), index=False, coerce_timestamps='ms')

        # On upload tous les fichiers dans le datastore
        datastore.upload(
            tmp_dir_name,
            target_path="clicks/",
            overwrite=True,
            show_progress=show_progress
        )


def get_articles(articles_metadata_path, articles_embeddings_path):
    """Renvoie un DataFrame contenant des informations sur les articles."""
    
    # On récupère les métadonnées des articles
    articles = pd.read_csv(articles_metadata_path)
    
    # On met à jour le type des variables
    articles = articles.astype({
        "article_id": np.uint64,
        "category_id": np.uint16,
        "created_at_ts": np.uint64,
        "publisher_id": np.uint8,
        "words_count": np.uint16
    })
    
    # On convertit les timestamps en datetime
    articles["created_at_ts"] = pd.to_datetime(articles['created_at_ts'], unit='ms')
    
    # On renomme les colonnes
    articles = articles.rename(columns={
        "created_at_ts": "created_dt",
        "words_count": "word_nb",
    })
    
    # On supprime les colonnes inutiles
    articles = articles.drop(columns=["publisher_id"])
    
    # On récupère les embeddings des articles
    with open(articles_embeddings_path, "rb") as f:
        articles_embeddings = pickle.load(f)
    
    # On expend les embeddings
    articles_embeddings = pd.DataFrame(
        articles_embeddings.tolist(),
        columns=[f"emb_{i}" for i in range(articles_embeddings.shape[1])]
    )
    
    # On ajoute les embeddings
    articles = pd.concat([articles, articles_embeddings], axis=1)
    
    # On trie par ordre chronologique
    articles = articles.sort_values("created_dt")
    
    return articles


def upload_articles_in_datastore(articles, datastore, show_progress=True):
    """"""
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        # On regroupe les données par an
        grp = articles.groupby(pd.Grouper(key="created_dt", freq="Y"))

        # On enregistre les dataframes
        for i, df in tqdm(grp, total=len(grp), leave=False):
            # On crée un dossier avec une arborescence de type date
            dir_name = os.path.join(tmp_dir_name, i.strftime("%Y"))
            os.makedirs(dir_name)

            # On enregistre les données. "coerce_timestamps" permet de conserver
            # le type datetime dans les métadonnées du fichier parquet.
            df.to_parquet(os.path.join(dir_name, "data.parquet"), index=False, coerce_timestamps='ms')

        # On upload tous les fichiers dans le datastore
        datastore.upload(
            tmp_dir_name,
            target_path="articles/",
            overwrite=True,
            show_progress=show_progress
        )


def filter_clicks(clicks, click_article_nb_ge=5):
    """"""
    
    # On groupe les données par utilisateur
    click_article_nbs = clicks.groupby("user_id").agg({"click_article_id": "nunique"})

    # On renomme les variables
    click_article_nbs = click_article_nbs.rename(columns={
        "click_article_id": "click_article_nb"
    })

    # On filtre les utilisateur par nombre d'article cliqués
    click_article_nbs = click_article_nbs[click_article_nbs["click_article_nb"] >= click_article_nb_ge]
    clicks = clicks[clicks["user_id"].isin(click_article_nbs.index)]

    return clicks


def get_user_article_ratings(clicks):
    """On renvoie un DataFrame contenant les notes attribués
    par les utilisateur aux articles sur lesquels ils ont cliqués.
    """
    
    # On filtre les variables
    user_article_ratings = clicks[["user_id", "click_dt"]].copy()
    user_article_ratings["article_id"] = clicks["click_article_id"]
    
    # On ajoute le nombre de click
    user_article_ratings["rating_click_nb"] = 1
    
    # On ajoute le nombre de click normalisé par le nombre total de clicks dans la session
    user_article_ratings["rating_click_per_session_ratio"] = 1 / clicks["session_size"]

    # On regoupe les couples utilisateur/articles
    user_article_ratings = user_article_ratings.groupby(["user_id", "article_id"], as_index=False)
    
    # On agrège les notes en les sommant
    user_article_ratings = user_article_ratings.agg({
        "rating_click_nb": "sum",
        "rating_click_per_session_ratio": "sum",
        "click_dt": "min"
    })

    # On trie dans l'ordre chronologique
    user_article_ratings = user_article_ratings.sort_values("click_dt")
    
    # On filtre les colonnes
    user_article_ratings = user_article_ratings[[
        "user_id",
        "article_id",
        "rating_click_nb",
        "rating_click_per_session_ratio",
        "click_dt"
    ]]
    
    return user_article_ratings

def add_user_article_ratings_tns(user_article_ratings, tn_nb=None, random_state=None):
    """On ajoute des vrais négatifs en prenant au hazard
    des articles non cliqués par l'utilisateur.
    """
    
    if tn_nb is None:
        # On crée un jeu de donnée équilibré
        tn_nb = np.ceil(
            len(user_article_ratings) / user_article_ratings["user_id"].nunique()
        ).astype(int)

    # On récupère la liste des articles du jeu de données
    article_ids = set(user_article_ratings["article_id"].unique())

    # On crée un générateur aléatoire
    rng = np.random.default_rng(random_state)
    
    # On regoupe les données par utilisateur
    grp = user_article_ratings.groupby("user_id")

    tns = []
    for user_id, df in tqdm(grp, total=len(grp), leave=False):
        # On crée une liste de "tn_nb" articles non cliqués par l'utilisateur
        user_article_ids = set(df["article_id"])
        tn_article_ids = list(article_ids - user_article_ids)
        tn_article_ids = rng.choice(tn_article_ids, tn_nb, replace=False)

        # On enregistre les vrais négatifs de l'utilisateur
        for article_id in tn_article_ids:
            tns.append({
                "user_id": user_id,
                "article_id": article_id,
                "rating_click_nb": 0,
                "rating_click_per_session_ratio": 0,
                "click_dt": pd.NaT,
            })

    # On concatène les données
    tns = pd.DataFrame(tns)
    user_article_ratings = pd.concat([user_article_ratings, tns], ignore_index=True)
    
    return user_article_ratings


def get_article_profiles(clicks, articles):
    """On renvoie un DataFrame représentant les profils des articles."""
    
    # On copie les données
    article_profiles = articles.copy()
    
    # On regroupe les données par article
    grp = clicks.groupby("click_article_id", as_index=False)

    # On aggrère les données
    grp_agg = grp.agg({
        "user_id": "count"
    })

    # On renomme les variables
    grp_agg = grp_agg.rename(columns={
        "user_id": "click_nb"
    })
    
    # On merge les données
    article_profiles = pd.merge(
        article_profiles,
        grp_agg,
        how="left",
        left_on="article_id",
        right_on="click_article_id"
    )
    
    # On remplace les valeurs indéfinies
    article_profiles = article_profiles.fillna(0)
    
    # On trie dans l'ordre chronologique
    article_profiles = article_profiles.sort_values("created_dt")
    
    # On réorganise les colonnes
    article_profiles = article_profiles[[
        "article_id",

        "created_dt",
        "category_id",
        "word_nb",

        "click_nb"
    ] + [f"emb_{i}" for i in range(250)]]
    
    return article_profiles

def get_user_profiles(clicks, article_profiles):
    """On renvoie un DataFrame représentant les profils des utilisateurs."""
    
    # On regroupe les données par utilisateur
    grp = pd.merge(clicks, article_profiles, how="left", left_on="click_article_id", right_on="article_id")
    grp = grp.groupby("user_id", as_index=False)

    # On aggrère les données
    user_profiles = grp.agg({
        "category_id": lambda x: x.mode()[0],
        "word_nb": "mean",
        "click_nb": "mean"
    })

    # On renomme les variables
    user_profiles = user_profiles.rename(columns={
        "category_id": "article_category_mode",
        "word_nb": "word_nb_mean",
        "click_nb": "click_nb_mean"
    })
    
    # On regroupe les données par utilisateur
    grp = clicks.groupby("user_id", as_index=False)

    grp_agg = []
    for i, df in tqdm(grp, leave=False):
        # On récupère la liste des tous les articles cliqués
        click_article_ids = df["click_article_id"].tolist()
        
        # On récupère les embeddings
        embeddings = article_profiles[article_profiles["article_id"].isin(click_article_ids)].iloc[:, -250:].values
            
        # On enregistre des statistiques
        grp_agg.append({
            "user_id": i,
            "article_embedding_mean": embeddings.mean(axis=0)
        })

    grp_agg = pd.DataFrame(grp_agg)
    
    # On merge les données
    user_profiles = pd.merge(user_profiles, grp_agg, on="user_id")
    
    # On expend les embeddings
    article_embedding_mean = pd.DataFrame(
        user_profiles["article_embedding_mean"].tolist(),
        columns=[f"emb_{i}" for i in range(250)]
    )
    
    # On ajoute les embeddings
    user_profiles = pd.concat([user_profiles, article_embedding_mean], axis=1)
    
    # On supprime les colonnes inutiles
    user_profiles = user_profiles.drop(columns=["article_embedding_mean"])
    
    return user_profiles
