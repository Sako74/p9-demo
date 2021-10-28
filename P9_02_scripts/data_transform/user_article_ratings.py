# -*- coding: utf-8 -*-
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse

from azureml.core import Run, Dataset, Datastore

from datasets import *
from utils import RANDOM_SEED

parser = argparse.ArgumentParser()

parser.add_argument("--clicks_path", type=str)
parser.add_argument("--tn_nb", type=lambda x: None if int(x) < 0 else int(x))
parser.add_argument("--dataset_name", type=str)

args = parser.parse_args()

print("On récupère le contexte, le workspace et le datastore par défaut.")

run = Run.get_context()
ws = run.experiment.workspace
datastore = ws.get_default_datastore()

print("On charge les datasets dans des DataFrame.")

clicks = pd.read_parquet(args.clicks_path)

print("On crée les notes des articles.")

user_article_ratings = get_user_article_ratings(clicks)

print("user_article_ratings.shape : ", user_article_ratings.shape)

print("On ajoute des vrais négatifs.")

user_article_ratings = add_user_article_ratings_tns(
    user_article_ratings,
    tn_nb=args.tn_nb,
    random_state=RANDOM_SEED
)

print("user_article_ratings.shape : ", user_article_ratings.shape)

print("On enregistre les données transformées.")

user_article_ratings = Dataset.Tabular.register_pandas_dataframe(
    dataframe=user_article_ratings,
    target=(datastore, args.dataset_name),
    name=args.dataset_name,
    description="Notations des articles"
)
