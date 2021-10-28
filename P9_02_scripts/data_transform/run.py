# -*- coding: utf-8 -*-
import os
import json

from azureml.core import Workspace, Dataset, Experiment, Environment, RunConfiguration

from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

from ..utils import *
from ..datasets import *


def get_default_step_ressources(ws, data_transform_path):
    """"""
    
    # On charge le cluster de calcul de type CPU si il existe ou on en crée un nouveau
    compute_target = get_compute_target(
        ws,
        name="p9-compute",
        location="northeurope",
        priority="lowpriority",
        vm_size="STANDARD_DS3_V2",
        max_nodes=1
    )
    
    # On spécifie les packages à installer
    env_conda = Environment.from_conda_specification(
        name="data_transform",
        file_path=os.path.join(data_transform_path, "conda_env.yml")
    )
    
    # On enregistre l'environnement
    env_conda.register(workspace=ws);
    
    # On crée la configuration d'exécution du script
    step_run_config = RunConfiguration()
    step_run_config.environment = env_conda
    
    return compute_target, step_run_config


def get_data_extraction_step(
        ws,
        data_transform_path,
        test_end_dt,
        test_day_nb,
        valid_day_nb,
        train_day_nb
    ):
    """"""
    
    # On récupère le dossier du script
    scripts_path, data_transform_dir = os.path.split(data_transform_path)
    
    # On récupère les ressources par dafaut
    compute_target, step_run_config = get_default_step_ressources(ws, data_transform_path)
    
    # On crée les liens qui vont permettre de lier les steps du pipeline
    test_clicks_path = PipelineData("test_clicks")
    valid_clicks_path = PipelineData("valid_clicks")
    train_clicks_path = PipelineData("train_clicks")
    articles_path = PipelineData("articles")
    
    # On crée l'étape du pipeline
    data_extraction_step = PythonScriptStep(
        name="data_extraction",
        source_directory=scripts_path,
        script_name=os.path.join(data_transform_dir, "data_extraction.py"),
        runconfig=step_run_config,
        compute_target=compute_target,
        arguments=[
            # Sorties
            "--test_clicks_path", test_clicks_path,
            "--valid_clicks_path", valid_clicks_path,
            "--train_clicks_path", train_clicks_path,
            "--articles_path", articles_path,
            
            # Paramètres
            "--test_end_dt", test_end_dt.isoformat(),
            "--test_day_nb", test_day_nb,
            "--valid_day_nb", valid_day_nb,
            "--train_day_nb", train_day_nb
        ],
        outputs=[
            test_clicks_path,
            valid_clicks_path,
            train_clicks_path,
            articles_path,
        ]
    )
    
    return data_extraction_step, test_clicks_path, valid_clicks_path, train_clicks_path, articles_path


def get_user_article_ratings_step(
        ws,
        data_transform_path,
        clicks_path,
        tn_nb,
        dataset_name
    ):
    """"""
    
    # On récupère le dossier du script
    scripts_path, data_transform_dir = os.path.split(data_transform_path)
    
    # On récupère les ressources par dafaut
    compute_target, step_run_config = get_default_step_ressources(ws, data_transform_path)
    
    # On crée l'étape du pipeline
    user_article_ratings_step = PythonScriptStep(
        name="user_article_ratings",
        source_directory=scripts_path,
        script_name=os.path.join(data_transform_dir, "user_article_ratings.py"),
        runconfig=step_run_config,
        compute_target=compute_target,
        arguments=[
            # Entrées
            "--clicks_path", clicks_path,
            
            # Paramètres
            "--tn_nb", -1 if tn_nb is None else tn_nb,
            "--dataset_name", dataset_name
        ],
        inputs=[clicks_path]
    )
    
    return user_article_ratings_step


def get_article_profiles_step(
        ws,
        data_transform_path,
        clicks_path,
        articles_path,
        dataset_name
    ):
    """"""
    
    # On récupère le dossier du script
    scripts_path, data_transform_dir = os.path.split(data_transform_path)
    
    # On récupère les ressources par dafaut
    compute_target, step_run_config = get_default_step_ressources(ws, data_transform_path)
    
    # On crée les liens qui vont permettre de lier les steps du pipeline
    article_profiles_path = PipelineData("article_profiles")
    
    # On crée l'étape du pipeline
    article_profiles_step = PythonScriptStep(
        name="article_profiles",
        source_directory=scripts_path,
        script_name=os.path.join(data_transform_dir, "article_profiles.py"),
        runconfig=step_run_config,
        compute_target=compute_target,
        arguments=[
            # Entrées
            "--clicks_path", clicks_path,
            "--articles_path", articles_path,
            
            # Sortie
            "--article_profiles_path", article_profiles_path,
            
            # Paramètres
            "--dataset_name", dataset_name
        ],
        inputs=[clicks_path, articles_path],
        outputs=[article_profiles_path]
    )
    
    return article_profiles_step, article_profiles_path


def get_user_profiles_step(
        ws,
        data_transform_path,
        clicks_path,
        article_profiles_path,
        dataset_name
    ):
    """"""
    
    # On récupère le dossier du script
    scripts_path, data_transform_dir = os.path.split(data_transform_path)
    
    # On récupère les ressources par dafaut
    compute_target, step_run_config = get_default_step_ressources(ws, data_transform_path)
    
    # On crée l'étape du pipeline
    user_profiles_step = PythonScriptStep(
        name="user_profiles",
        source_directory=scripts_path,
        script_name=os.path.join(data_transform_dir, "user_profiles.py"),
        runconfig=step_run_config,
        compute_target=compute_target,
        arguments=[
            # Entrées
            "--clicks_path", clicks_path,
            "--article_profiles_path", article_profiles_path,
            
            # Paramètres
            "--dataset_name", dataset_name
        ],
        inputs=[clicks_path, article_profiles_path]
    )
    
    return user_profiles_step


def exp_submit(ws, steps, regenerate_outputs=True, wait_for_completion=False, show_output=False):
    """"""
    
    # On crée le pipeline
    pipeline = Pipeline(workspace=ws, steps=steps)
    
    # On valide le pipeline
    errors = pipeline.validate()
    assert len(errors) == 0, print(errors)
    
    # On crée une expérience
    exp = Experiment(workspace=ws, name="data_transform")

    # On lance l'exécution du script
    run = exp.submit(pipeline, regenerate_outputs=regenerate_outputs)
    
    if wait_for_completion:
        # On attend la fin de l'exécution
        run.wait_for_completion(show_output=show_output)
    
    return run


if __name__ == "__main__":
    # On charge les variables d'environnemnt
    azure_credentials = json.loads(os.getenv("AZURE_CREDENTIALS"))
    azure_workspace = json.loads(os.getenv("AZURE_WORKSPACE"))
    
    # On charge l’espace de travail Azure Machine Learning existant
    ws = get_ws(azure_credentials, azure_workspace)
    
    # On crée le chemin du dossier des scripts de transformation des données
    data_transform_path = SCRIPTS_PATH + "data_transform"
    
    # On crée l'étape d'extraction des données du feature store
    (
        data_extraction_step,
        test_clicks_path,
        valid_clicks_path,
        train_clicks_path,
        articles_path
    ) = get_data_extraction_step(
        ws,
        data_transform_path,
        test_end_dt=datetime(2017, 10, 17),
        test_day_nb=1,
        valid_day_nb=1,
        train_day_nb=5
    )
    
    # On crée les étapes de notation des articles
    test_user_article_ratings_step = get_user_article_ratings_step(
        ws,
        data_transform_path,
        clicks_path=test_clicks_path,
        tn_nb=100,
        dataset_name="test_user_article_ratings"
    )

    valid_user_article_ratings_step = get_user_article_ratings_step(
        ws,
        data_transform_path,
        clicks_path=valid_clicks_path,
        tn_nb=100,
        dataset_name="valid_user_article_ratings"
    )

    train_user_article_ratings_step = get_user_article_ratings_step(
        ws,
        data_transform_path,
        clicks_path=train_clicks_path,
        tn_nb=None,
        dataset_name="train_user_article_ratings"
    )
    
    # On crée l'étape de création des profils des articles
    article_profiles_step, article_profiles_path = get_article_profiles_step(
        ws,
        data_transform_path,
        clicks_path=train_clicks_path,
        articles_path=articles_path,
        dataset_name="article_profiles"
    )
    
    # On crée l'étape de création des profils des utilisateurs
    train_user_profiles_step = get_user_profiles_step(
        ws,
        data_transform_path,
        clicks_path=train_clicks_path,
        article_profiles_path=article_profiles_path,
        dataset_name="train_user_profiles"
    )
    
    # On soummet l'exécution du pipeline
    run = exp_submit(
        ws,
        steps=[
            data_extraction_step,
            test_user_article_ratings_step,
            valid_user_article_ratings_step,
            train_user_article_ratings_step,
            article_profiles_step,
            train_user_profiles_step
        ],
        regenerate_outputs=True,
        wait_for_completion=True,
        show_output=True
    )
