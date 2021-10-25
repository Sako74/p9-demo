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
    Model, Run, Experiment, ScriptRunConfig
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
MODEL_PATH = "data/model/"
SCRIPTS_PATH = "scripts/"

os.makedirs(CSV_PATH, exist_ok=True)
os.makedirs(PICKLE_PATH, exist_ok=True)
os.makedirs(PARQUET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

RANDOM_SEED = 42
MAX_TOTAL_RUNS = 50


def get_compute_target(ws, name, location, priority, vm_size, max_nodes=1):
    """"""
    try:
        # On charge le cluster de calcul si il existe
        compute_target = ComputeTarget(workspace=ws, name=name)
    except ComputeTargetException:
        # Si le cluster n'existe pas, on on en crée un nouveau
        config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_NC6",
            location=location,
            vm_priority="lowpriority",
            min_nodes=0,
            max_nodes=1
        )

        compute_target = ComputeTarget.create(
            workspace=ws,
            name=name,
            provisioning_configuration=config
        )
        
        compute_target.wait_for_completion(
            show_output=True,
            min_node_count=None,
            timeout_in_minutes=20
        )
        
    return compute_target


def clear_datastore(datastore):
    """Supprime tout le contenu du datastore."""
    for i in datastore.blob_service.list_blobs(datastore.container_name):
        datastore.blob_service.delete_blob(datastore.container_name, i.name)


def create_update_clicks_dataset(ws, datastore):
    """"""
    # On crée un dataset avec tous les fichiers clicks
    clicks_ds = Dataset.Tabular.from_parquet_files(path=(datastore, "clicks/**/data.parquet"))

    # On spécifie la colonne datetime qui va permettre de filtrer les données
    clicks_ds = clicks_ds.with_timestamp_columns(timestamp="click_dt")

    # On crée/update le dataset
    clicks_ds = clicks_ds.register(ws, name="clicks", create_new_version=True)
    
    return clicks_ds


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


def create_update_articles_dataset(ws, datastore):
    """"""
    # On crée un dataset avec tous les fichiers articles
    articles_ds = Dataset.Tabular.from_parquet_files(path=(datastore, "articles/**/data.parquet"))

    # On spécifie la colonne datetime qui va permettre de filtrer les données
    articles_ds = articles_ds.with_timestamp_columns(timestamp="created_dt")

    # On crée/update le dataset
    articles_ds = articles_ds.register(ws, name="articles", create_new_version=True)
    
    return articles_ds


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


def get_last_run(ws, exp_name):
    """Renvoie la dernière exécution de l'expérience spécifiée"""
    # On récupère l'exécution de la dernière expérience.
    # Ce code a été ajouté en cas de coupure de connexion avec le notebook...
    exp = Experiment(workspace=ws, name=exp_name)
    
    # On renvoie la dernière exécution
    return next(exp.get_runs())
