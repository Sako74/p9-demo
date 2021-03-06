# -*- coding: utf-8 -*-
import os
import json

from azureml.core import Workspace, Dataset, Experiment, Environment, ScriptRunConfig

from azureml.train.hyperdrive import HyperDriveConfig, GridParameterSampling, BanditPolicy
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice, uniform

from ..utils import *
from ..models import *


def exp_submit(
        ws,
        model_train_path,
        params=None,
        gs_params=None,
        wait_for_completion=False,
        show_output=False
    ):
    """Lance l'exécution de l'entrainement du modèle.
    
    Parameters
    ----------
        ws : Azure Workspace
            Workspace de Azure ML.
        model_train_path : str
            Dossier contenant les scripts d'entrainement du modèle.
        params : dict
            Paramètres du modèle.
        gs_params : dict
            Hyperparamètres à tester.
        wait_for_completion : bool
            Attendre la fin de l'entrainement du modèle.
        show_output : bool
            Afficher les logs de sortie.

    Returns
    ----------
        Azure Run
            Exécution de l'entrainement du modèle.
    """
    
    # On crée le nom de l'expérience à partir du nom du dossier
    scripts_path, exp_name = os.path.split(model_train_path)

    # On charge le cluster de calcul de type CPU si il existe ou on en crée un nouveau
    compute_target = get_compute_target(
        ws,
        name="p9-compute",
        location="northeurope",
        priority="lowpriority",
        vm_size="STANDARD_DS3_V2",
        max_nodes=5
    )

    # On spécifie les packages à installer
    env_conda = Environment.from_conda_specification(
        name=exp_name,
        file_path=os.path.join(model_train_path, "conda_env.yml")
    )

    # On enregistre l'environnement
    env_conda.register(workspace=ws);
    
    # On crée une expérience
    exp = Experiment(workspace=ws, name=exp_name)
    
    if params is None:
        # On charge les paramètres
        with open(os.path.join(model_train_path, "params.json")) as f:
            params = json.load(f)
    
    # On charge les datasets
    train_user_article_ratings_ds = Dataset.get_by_name(ws, params.get("train_user_article_ratings"))
    valid_user_article_ratings_ds = Dataset.get_by_name(ws, params.get("valid_user_article_ratings"))
    article_profiles_ds = Dataset.get_by_name(ws, params.get("article_profiles"))
    train_user_profiles_ds = Dataset.get_by_name(ws, params.get("train_user_profiles"))
    
    # On définit les paramètres de l'expérience
    args = [
        # Jeux de données
        "--train_user_article_ratings_id", train_user_article_ratings_ds.as_named_input("train_user_article_ratings_ds"),
        "--valid_user_article_ratings_id", valid_user_article_ratings_ds.as_named_input("valid_user_article_ratings_ds"),
        "--article_profiles_id", article_profiles_ds.as_named_input("article_profiles_ds"),
        "--train_user_profiles_id", train_user_profiles_ds.as_named_input("train_user_profiles_ds")
    ]
    
    # On ajoute les hyperparamètres si ils sont présents
    hyperparameters = [
        "rating_col",
        "num_vars_scale",
        "cat_vars_scale"
    ]
    for i in hyperparameters:
        if i in params:
            args += [f"--{i}", params.get(i)]
    
    # On crée la configuration d'exécution du script d'entrainement
    src = ScriptRunConfig(
        compute_target=compute_target,
        environment=env_conda,
        source_directory=scripts_path,
        script=os.path.join(exp_name, "train.py"),
        arguments=args
    )
    
    if gs_params is not None:
        # On liste les hyperparamètres que l'on souhaite tester
        param_sampling = GridParameterSampling({f"--{k}": choice(v) for k, v in gs_params.items()})

        # On crée la configuration pour la recherche des meilleurs hyperparamètres
        hyperdrive_config = HyperDriveConfig(
            run_config=src,
            hyperparameter_sampling=param_sampling,
            primary_metric_name=PRIMARY_METRIC_NAME,
            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
            max_total_runs=50
        )
        # On lance l'exécution de la recherche des hyperparamètres
        run = exp.submit(hyperdrive_config)
    else:
        # On lance l'exécution de l'entrainement
        run = exp.submit(src)
    
    if wait_for_completion:
        # On attend la fin de l'exécution
        run.wait_for_completion(show_output=show_output);
    
    return run


if __name__ == "__main__":
    # On charge les variables d'environnemnt
    azure_credentials = json.loads(os.getenv("AZURE_CREDENTIALS"))
    azure_workspace = json.loads(os.getenv("AZURE_WORKSPACE"))
    
    # On charge l’espace de travail Azure Machine Learning existant
    ws = get_ws(azure_credentials, azure_workspace)
    
    # On soummet l'exécution de l'expérience
    run = exp_submit(
        ws,
        "P9_02_scripts/model_content_based_train",
        params=None,
        gs_params=None,
        wait_for_completion=True,
        show_output=True
    )
    
    # On enregistre le modèle
    model = run.register_model(
        model_name="recommender",
        model_path="outputs/model.joblib",
        tags={"class_name": "ContentBasedRecommender"}
    )
