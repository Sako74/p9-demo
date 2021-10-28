# -*- coding: utf-8 -*-
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
from datetime import datetime, date, timedelta

from azureml.core import Run, Dataset, Datastore

from datasets import *


def get_clicks_dataset(ws, start_time=None, end_time=None, include_boundary=True):
    """"""
    # On récupère le dataset
    clicks_ds = Dataset.get_by_name(ws, "clicks")
    
    # On filtre les données en fonction de l'horodatage des clicks
    if start_time and end_time:
        clicks_ds = clicks_ds.time_between(start_time, end_time, include_boundary=include_boundary)
    elif start_time:
        clicks_ds = clicks_ds.time_after(start_time, include_boundary=include_boundary)
    elif end_time:
        clicks_ds = clicks_ds.time_before(end_time, include_boundary=include_boundary)
    
    return clicks_ds


def get_articles_dataset(ws, start_time=None, end_time=None, include_boundary=True):
    """"""
    # On récupère le dataset
    articles_ds = Dataset.get_by_name(ws, "articles")
    
    # On filtre les données en fonction de l'horodatage des articles
    if start_time and end_time:
        articles_ds = articles_ds.time_between(start_time, end_time, include_boundary=include_boundary)
    elif start_time:
        articles_ds = articles_ds.time_after(start_time, include_boundary=include_boundary)
    elif end_time:
        articles_ds = articles_ds.time_before(end_time, include_boundary=include_boundary)
    
    return articles_ds


parser = argparse.ArgumentParser()

# +
parser.add_argument("--test_clicks_path", type=str)
parser.add_argument("--valid_clicks_path", type=str)
parser.add_argument("--train_clicks_path", type=str)
parser.add_argument("--articles_path", type=str)

# Date par défaut du dernier jour du jeu de test
yesterday = date.today() - timedelta(days=1)

parser.add_argument("--test_end_dt", type=datetime.fromisoformat, default=yesterday)

parser.add_argument("--test_day_nb", type=int)
parser.add_argument("--valid_day_nb", type=int)
parser.add_argument("--train_day_nb", type=int)
# -

args = parser.parse_args()

print("On récupère le contexte, le workspace et le datastore par défaut.")

run = Run.get_context()
ws = run.experiment.workspace
datastore = ws.get_default_datastore()

print("On effectue un split temporel les données.")

# +
test_end_dt = args.test_end_dt
test_start_dt = test_end_dt - timedelta(days=args.test_day_nb)

valid_end_dt = test_start_dt - timedelta(days=1)
valid_start_dt = valid_end_dt - timedelta(days=args.valid_day_nb)

train_end_dt = valid_start_dt - timedelta(days=1)
train_start_dt = train_end_dt - timedelta(days=args.train_day_nb)

print("Test period :\t", test_end_dt, "-", test_start_dt)
print("Valid period :\t", valid_end_dt, "-", valid_start_dt)
print("Train period :\t", train_end_dt, "-", train_start_dt)
# -

print("On récupère le dataset filtré en fonction de l'horodatage des clicks.")

test_clicks_ds = get_clicks_dataset(ws, start_time=test_start_dt, end_time=test_end_dt)
valid_clicks_ds = get_clicks_dataset(ws, start_time=valid_start_dt, end_time=valid_end_dt)
train_clicks_ds = get_clicks_dataset(ws, start_time=train_start_dt, end_time=train_end_dt)

print("On mets les données dans des dataframes.")

test_clicks = test_clicks_ds.to_pandas_dataframe().reset_index(drop=True)
valid_clicks = valid_clicks_ds.to_pandas_dataframe().reset_index(drop=True)
train_clicks = train_clicks_ds.to_pandas_dataframe().reset_index(drop=True)

print("Test dataset shape :\t", test_clicks.shape)
print("Valid dataset shape :\t", valid_clicks.shape)
print("Train dataset shape :\t", train_clicks.shape)

print("On récupère les informations des articles.")

articles_ds = get_articles_dataset(ws)

print("On mets les données dans un dataframes.")

articles = articles_ds.to_pandas_dataframe().reset_index(drop=True)

print("Articles dataset shape :\t", articles.shape)

print("On filtre les utilisateurs ayant effectué trop peu de clicks.")

test_clicks = filter_clicks(test_clicks, click_article_nb_ge=5)
valid_clicks = filter_clicks(valid_clicks, click_article_nb_ge=5)

print("On calcule le nombre d'utilisateurs.")

test_user_ids = set(test_clicks["user_id"].unique())
valid_user_ids = set(valid_clicks["user_id"].unique())
train_user_ids = set(train_clicks["user_id"].unique())

for name, user_ids in zip(["test", "validation"], [test_user_ids, valid_user_ids]):
    user_nb = len(test_user_ids)

    known_user_nb = len(user_ids & train_user_ids)
    known_user_ratio = known_user_nb / len(user_ids)

    unknown_user_nb = len(user_ids - train_user_ids)
    unknown_user_ratio = unknown_user_nb / len(user_ids)

    print(f"Le jeu de {name} contient :")
    print(f"- {known_user_ratio:.1%} ({known_user_nb}/{user_nb}) utilisateurs connus.")
    print(f"- {unknown_user_ratio:.1%} ({unknown_user_nb}/{user_nb}) utilisateurs inconnus.\n")

print("On renvoie les dataframes.")

test_clicks.to_parquet(
    args.test_clicks_path,
    index=False,
    coerce_timestamps="ms"
)

valid_clicks.to_parquet(
    args.valid_clicks_path,
    index=False,
    coerce_timestamps="ms"
)

train_clicks.to_parquet(
    args.train_clicks_path,
    index=False,
    coerce_timestamps="ms"
)

articles.to_parquet(
    args.articles_path,
    index=False,
    coerce_timestamps="ms"
)
