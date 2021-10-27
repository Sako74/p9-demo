# -*- coding: utf-8 -*-
import os
import json

from azureml.core import Model, Environment

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

from ..utils import *
from ..models import *


def model_deploy(
        ws,
        model_deploy_path,
        wait_for_deployment=False,
        show_output=False
    ):
    """"""
    
    # On crée le nom de l'expérience à partir du nom du dossier
    scripts_path, model_deploy_dir = os.path.split(model_deploy_path)
    
    # On récupère notre modèle
    model = Model(ws, "recommender")
    
    # On spécifie les packages à installer
    env_conda = Environment.from_conda_specification(
        name=model_deploy_dir,
        file_path=os.path.join(model_deploy_path, "conda_env.yml")
    )

    # On enregistre l'environnement
    env_conda.register(workspace=ws);
    
    # On crée la configuration pour le déploiement.
    aciconfig = AciWebservice.deploy_configuration(
        cpu_cores=1, 
        memory_gb=8,
        description="Recommendation d'articles"
    )

    # On spécifie l'environnement et le script chargé de prédire
    # si un tweet est négatif ou non.
    inference_config = InferenceConfig(
        source_directory=scripts_path,
        entry_script=os.path.join(model_deploy_dir, "score.py"),
        environment=env_conda
    )
    
    # On déploie le modèle
    model_aci = Model.deploy(
        workspace=ws, 
        name="p9-recommender-aci",
        models=[model], 
        inference_config=inference_config, 
        deployment_config=aciconfig
    )
    
    if wait_for_deployment:
        # On attend la fin du déploiement
        model_aci.wait_for_deployment(show_output=show_output)
        
    return model_aci


if __name__ == "__main__":
    # On charge les variables d'environnemnt
    azure_credentials = json.loads(os.getenv("AZURE_CREDENTIALS"))
    azure_workspace = json.loads(os.getenv("AZURE_WORKSPACE"))
    
    # On charge l’espace de travail Azure Machine Learning existant
    ws = get_ws(azure_credentials, azure_workspace)
    
    # On déploie le modèle
    model_aci = model_deploy(
        ws,
        "P9_02_scripts/model_content_based_deploy_aci",
        wait_for_deployment=True,
        show_output=True
    )
    
    # On affiche l'URL de l'API
    print(model_aci.scoring_uri)
