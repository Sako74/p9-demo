# -*- coding: utf-8 -*-
import os
import gc
import json
import requests
from shutil import copy2
from configparser import ConfigParser

import matplotlib.pyplot as plt
import seaborn as sns

from azureml.core import (
    Workspace, Dataset, Datastore, Environment,
    Model, Run, Experiment, ScriptRunConfig, Webservice
)
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.compute import AmlCompute, ComputeTarget

from azureml.exceptions import ComputeTargetException

from azureml.train.hyperdrive import HyperDriveConfig, GridParameterSampling, BanditPolicy
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice, uniform

from azureml.widgets import RunDetails

from tqdm.notebook import tqdm

from IPython.display import display


import dotenv

# On définit les variables globales
CSV_PATH = "data/csv/"
PICKLE_PATH = "data/pickle/"
PARQUET_PATH = "data/parquet/"
PICKLE_PATH = "data/pickle/"
MODEL_PATH = "data/model/"
SCRIPTS_PATH = "../P9_02_scripts/"
FUNCTION_PATH = "../P9_03_function/"

os.makedirs(CSV_PATH, exist_ok=True)
os.makedirs(PICKLE_PATH, exist_ok=True)
os.makedirs(PARQUET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

RANDOM_SEED = 42
MAX_TOTAL_RUNS = 50


def clear_datastore(datastore):
    """Supprime tout le contenu d'un Blob Datastore.
    
    Parameters
    ----------
        datastore : Azure Blob Datastore
            Blob Datastore Azure.
    """
    
    for i in datastore.blob_service.list_blobs(datastore.container_name):
        datastore.blob_service.delete_blob(datastore.container_name, i.name)


def create_update_clicks_dataset(ws, datastore):
    """Crée/update le Dataset clicks avec tous les
    fichiers clicks présents sur le datatstore.
    
    Parameters
    ----------
        ws : Azure Workspace
            Workspace de Azure ML.
        datastore : Azure Blob Datastore
            Blob Datastore Azure.

    Returns
    ----------
        Azure Dataset
            Dataset clicks.
    """
    
    # On crée un dataset avec tous les fichiers clicks
    clicks_ds = Dataset.Tabular.from_parquet_files(path=(datastore, "clicks/**/data.parquet"))

    # On spécifie la colonne datetime qui va permettre de filtrer les données
    clicks_ds = clicks_ds.with_timestamp_columns(timestamp="click_dt")

    # On crée/update le dataset
    clicks_ds = clicks_ds.register(ws, name="clicks", create_new_version=True)
    
    return clicks_ds


def create_update_articles_dataset(ws, datastore):
    """Crée/update le Dataset articles avec tous les
    fichiers articles présents sur le datatstore.
    
    Parameters
    ----------
        ws : Azure Workspace
            Workspace de Azure ML.
        datastore : Azure Blob Datastore
            Blob Datastore Azure.

    Returns
    ----------
        Azure Dataset
            Dataset articles.
    """
    
    # On crée un dataset avec tous les fichiers articles
    articles_ds = Dataset.Tabular.from_parquet_files(path=(datastore, "articles/**/data.parquet"))

    # On spécifie la colonne datetime qui va permettre de filtrer les données
    articles_ds = articles_ds.with_timestamp_columns(timestamp="created_dt")

    # On crée/update le dataset
    articles_ds = articles_ds.register(ws, name="articles", create_new_version=True)
    
    return articles_ds


def get_last_run(ws, exp_name):
    """Renvoie la dernière exécution de l'expérience spécifiée.
    
    Parameters
    ----------
        ws : Azure Workspace
            Workspace de Azure ML.
        exp_name : str
            Nom de l'expérience.

    Returns
    ----------
        Azure Run
            Dernière exécution de l'expérience.
    """
    
    # On récupère l'exécution de la dernière expérience.
    # Ce code a été ajouté en cas de coupure de connexion avec le notebook...
    exp = Experiment(workspace=ws, name=exp_name)
    
    # On renvoie la dernière exécution
    return next(exp.get_runs())
