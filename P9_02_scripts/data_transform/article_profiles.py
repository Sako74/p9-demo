# -*- coding: utf-8 -*-
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse

from azureml.core import Run, Dataset, Datastore

from datasets import *

parser = argparse.ArgumentParser()

parser.add_argument("--clicks_path", type=str)
parser.add_argument("--articles_path", type=str)
parser.add_argument("--article_profiles_path", type=str)
parser.add_argument("--dataset_name", type=str)

args = parser.parse_args()

print("On récupère le contexte, le workspace et le datastore par défaut.")

run = Run.get_context()
ws = run.experiment.workspace
datastore = ws.get_default_datastore()

print("On charge les datasets dans des DataFrame.")

clicks = pd.read_parquet(args.clicks_path)
articles = pd.read_parquet(args.articles_path)

print("On crée les profils des articles.")

article_profiles = get_article_profiles(clicks, articles)

print("article_profiles.shape : ", article_profiles.shape)

print("On enregistre les données transformées.")

article_profiles_ds = Dataset.Tabular.register_pandas_dataframe(
    dataframe=article_profiles,
    target=(datastore, args.dataset_name),
    name=args.dataset_name,
    description="Profils des articles"
)

print("On renvoie le dataframe.")

article_profiles.to_parquet(
    args.article_profiles_path,
    index=False,
    coerce_timestamps="ms"
)
