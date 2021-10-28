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
parser.add_argument("--article_profiles_path", type=str)
parser.add_argument("--dataset_name", type=str)

args = parser.parse_args()

print("On récupère le contexte, le workspace et le datastore par défaut.")

run = Run.get_context()
ws = run.experiment.workspace
datastore = ws.get_default_datastore()

print("On charge les datasets dans des DataFrame.")

clicks = pd.read_parquet(args.clicks_path)
article_profiles = pd.read_parquet(args.article_profiles_path)

print("On crée les profils des utilisateurs.")

user_profiles = get_user_profiles(clicks, article_profiles)

print("user_profiles.shape : ", user_profiles.shape)

print("On enregistre les données transformées.")

user_profiles = Dataset.Tabular.register_pandas_dataframe(
    dataframe=user_profiles,
    target=(datastore, args.dataset_name),
    name=args.dataset_name,
    description="Profils des utilisateurs"
)
