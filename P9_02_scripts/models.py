# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import joblib

from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

import surprise

PRIMARY_METRIC_NAME = "recall@5"


def get_precision_recall_n_score(model, model_name, user_article_ratings, article_profiles, top_n=5):
    """"""
    
    # On récupère les vrais positifs et les vrais négatifs
    tp = user_article_ratings[user_article_ratings["rating"] > 0]
    tn = user_article_ratings[user_article_ratings["rating"] == 0]

    # On les regoupe par utilisateur
    tp_grp = tp.groupby("user_id", as_index=False)
    tn_grp = tn.groupby("user_id", as_index=False)

    # On agrège les ids des articles dans des listes
    tp = tp_grp.agg({"article_id": list})
    tp = tp.rename(columns={"article_id": "click_article_ids"})

    tn = tn_grp.agg({"article_id": list})
    tn = tn.rename(columns={"article_id": "no_click_article_ids"})

    # On rassemble les vrais positifs et les vrais négatifs mis dans des listes
    test_ds = pd.merge(tp, tn)
    
    precision_n_score = []
    recall_n_score = []
    for _, user_row in tqdm(test_ds.iterrows(), total=len(test_ds), leave=False):
        
        # On demande une recommandation parmis les articles non 
        # cliqués (no_click_article_ids) et les articles cliqués (article_id).
        article_recommanded_ids = model.recommand_from_no_click_article_ids(
            user_row["user_id"],
            user_row["click_article_ids"] + user_row["no_click_article_ids"]
        )
        
        # On calcule la précision 
        precision = len(set(article_recommanded_ids[:top_n]) & set(user_row["click_article_ids"])) / top_n

        # On ajoute le score de l'utilisateur
        precision_n_score.append(precision)

        user_recall_n_score = []
        for article_id in user_row["click_article_ids"]:
            exclude = user_row["click_article_ids"]
            exclude.remove(article_id)
            article_recommanded_ids_tmp = [i for i in article_recommanded_ids if i not in exclude]
            
            # On regarde si l'article est bien dans les recommandations
            recall = int(article_id in article_recommanded_ids_tmp[:top_n])
            user_recall_n_score.append(recall)
            
        # On calcule le score moyen de l'utilisateur
        user_recall_n_score = np.mean(user_recall_n_score)
            
        # On ajoute le score de l'utilisateur
        recall_n_score.append(user_recall_n_score)
        
    # On calcule les scores moyen
    precision_n_score = np.mean(precision_n_score)
    recall_n_score = np.mean(recall_n_score)
    
    # On renvoie un DataFrame avec les résultats
    res = {
        "model": model_name,
        f"precision@{top_n}": precision_n_score,
        f"recall@{top_n}": recall_n_score
    }
    res = pd.DataFrame([res])
    
    return res

def get_no_click_article_ids(user_id, session_start_dt, article_profiles, user_article_ratings):
    """"""
    # On récupère la liste des articles parus avant la session de l'utilisateur
    no_click_article_ids = article_profiles[article_profiles["created_dt"] <= session_start_dt]
    no_click_article_ids = set(no_click_article_ids["article_id"])
    
    # On récupère la liste des articles déjà cliqués par l'utilisateur
    already_clicked_ids = user_article_ratings[user_article_ratings["user_id"] == user_id]
    already_clicked_ids = set(already_clicked_ids["article_id"])
    
    # On ne conserve que les articles qui n'ont pas encore été cliqués par l'utilisateur
    no_click_article_ids = list(no_click_article_ids - already_clicked_ids)
    
    return no_click_article_ids


class Recommender(ABC):
    """"""
    
    def __init__(self, article_profiles, user_article_ratings):
        self.article_profiles = article_profiles
        self.user_article_ratings = user_article_ratings
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def recommand_from_no_click_article_ids(self, user_id, no_click_article_ids, top_n=None, return_ratings=False):
        pass
    
    def recommand(self, user_id, session_start_dt, top_n=None):
        # On récupère les profils des articles non cliqués par l'utilisateur
        no_click_article_ids = get_no_click_article_ids(
            user_id,
            session_start_dt,
            self.article_profiles,
            self.user_article_ratings
        )
        
        # On renvoie les recommandations
        return self.recommand_from_no_click_article_ids(
            user_id,
            no_click_article_ids,
            top_n=top_n
        )

class MostRecentRecommender(Recommender):
    """"""
    
    def __init__(self, article_profiles, user_article_ratings):
        super().__init__(article_profiles, user_article_ratings)
        
    def fit(self):
        pass
    
    def recommand_from_no_click_article_ids(self, user_id, no_click_article_ids, top_n=None, return_ratings=False): 
        # On récupère les profiles des articles
        no_click_article_profiles = self.article_profiles[
            self.article_profiles["article_id"].isin(no_click_article_ids)
        ]
        
        # On classe les articles par récence et ensuite par nombre de clicks
        no_click_article_profiles = no_click_article_profiles.sort_values(
            by=["created_dt", "click_nb"],
            ascending=True
        )
        
        best_article_ids = no_click_article_profiles["article_id"].values
    
        if return_ratings:
            ratings = np.linspace(1, 0, len(best_article_ids))
            return best_article_ids[:top_n], ratings[:top_n]
        else:
            return best_article_ids[:top_n]

class ContentBasedRecommender(Recommender):
    """"""
    
    def __init__(
            self,
            article_profiles,
            user_article_ratings,
            user_profiles,
            num_vars_scale=0,
            cat_vars_scale=0
        ):
        super().__init__(article_profiles, user_article_ratings)

        self.user_profiles = user_profiles
        self.num_vars_scale = num_vars_scale
        self.cat_vars_scale = cat_vars_scale
        
        self.cold_start_model = MostRecentRecommender(article_profiles, user_article_ratings)
        
    def fit(self):
        # On crée l'encodeur des variables de type catégorie
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder.fit(self.article_profiles[["category_id"]].values)

        # On crée un scaler pour les variables numériques
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.scaler.fit(self.article_profiles[["word_nb", "click_nb"]].values)
        
    def get_user_features(self, user_id):
        # On récupère le profil de l'utilisateur
        user_profile = self.user_profiles[self.user_profiles["user_id"] == user_id]

        # On encode les variables de type catégorie
        features = self.encoder.transform(user_profile[["article_category_mode"]].values) * self.cat_vars_scale

        # On scale les variables de type numérique et on concatène les features
        features = np.hstack([
            features,
            self.scaler.transform(user_profile[["word_nb_mean", "click_nb_mean"]].values) * self.num_vars_scale,
            user_profile.iloc[:, -250:].values
        ])

        return features.flatten()
    
    def get_article_features_and_ids(self, article_ids):
        # On récupère les profils des articles
        article_profiles = self.article_profiles[
            self.article_profiles["article_id"].isin(article_ids)
        ]

        # On encode les variables de type catégorie
        features = self.encoder.transform(article_profiles[["category_id"]].values) * self.cat_vars_scale

        # On scale les variables de type numérique et on concatène les features
        features = np.hstack([
            features,
            self.scaler.transform(article_profiles[["word_nb", "click_nb"]].values) * self.num_vars_scale,
            article_profiles.iloc[:, -250:].values
        ])

        return features, article_profiles["article_id"].values
    
    def recommand_from_no_click_article_ids(self, user_id, no_click_article_ids, top_n=None, return_ratings=False):
        # On vérifie si l'utilisateur est nouveau (cold start problem)
        if user_id not in self.user_profiles["user_id"].values:
            return self.cold_start_model.recommand_from_no_click_article_ids(
                user_id,
                no_click_article_ids,
                top_n=top_n,
                return_ratings=return_ratings
            )
        
        # On récupère les features du profil de l'utilisateur
        user_features = self.get_user_features(user_id)
        
        # On récupère les features des profils des articles
        article_features, article_ids = self.get_article_features_and_ids(no_click_article_ids)
        
        # On calcule les scores de cosine similarity
        cosine_similarities = cosine_similarity(user_features[None,], article_features)
        
        # On trie les articles par meilleur score de similarité
        cosine_similarities = cosine_similarities.flatten()
        best_cosine_similarities_ids = np.argsort(cosine_similarities)[::-1]
        best_article_ids = article_ids[best_cosine_similarities_ids]
        
        if return_ratings:
            best_cosine_similarities = cosine_similarities[best_cosine_similarities_ids]
            return best_article_ids[:top_n], best_cosine_similarities[:top_n]
        else:
            return best_article_ids[:top_n]


class CollaborativeRecommender(Recommender):
    """"""
    
    def __init__(self, article_profiles, user_article_ratings, **kwargs):
        super().__init__(article_profiles, user_article_ratings)
        
        self.algo = surprise.SVD(**kwargs)
        
        self.cold_start_model = MostRecentRecommender(article_profiles, user_article_ratings)
        
    def fit(self):
        # On copie les notes
        ratings = self.user_article_ratings.copy()
        
        # On calcule les notes min et max en filtrant les outliers
        rating_min = ratings["rating"].min()
        rating_max = ratings["rating"].quantile(0.99)
        
        # On normalise les notes entre 0 et 1
        ratings["rating"] = ratings["rating"].clip(rating_min, rating_max) / (rating_max - rating_min)
        
        # On construit un dataset Surprise pour entrainer le modèle
        self.reader = surprise.Reader(rating_scale=(
            rating_min,
            rating_max
        ))
        self.trainset = surprise.Dataset.load_from_df(ratings, self.reader)
        self.trainset = self.trainset.build_full_trainset()
        
        # On entraine le modèle
        self.algo.fit(self.trainset)
    
    def recommand_from_no_click_article_ids(self, user_id, no_click_article_ids, top_n=None, return_ratings=False):
        # On vérifie si l'utilisateur est nouveau (cold start problem)
        if not self.trainset.knows_user(user_id):
            return self.cold_start_model.recommand_from_no_click_article_ids(
                user_id,
                no_click_article_ids,
                top_n=top_n,
                return_ratings=return_ratings
            )
        
        # On crée un jeu de test
        df = pd.DataFrame({
            "user_id": [user_id] * len(no_click_article_ids),
            "article_id": no_click_article_ids,
            "rating": [0] * len(no_click_article_ids)
        })
        
        # On construit un dataset Surprise pour tester le modèle
        ds = surprise.Dataset.load_from_df(df, self.reader)
        ds = ds.build_full_trainset().build_testset()
        
        # On demande au modèle de prédire les ratings
        preds = self.algo.test(ds)
        preds = np.array([i.est for i in preds])
        
        # On trie les articles par meilleur ratings
        best_preds_ids = np.argsort(preds)[::-1]
        best_article_ids = df["article_id"].values[best_preds_ids]
        
        if return_ratings:
            best_preds = preds[best_preds_ids]
            return best_article_ids[:top_n], best_preds[:top_n]
        else:
            return best_article_ids[:top_n]
