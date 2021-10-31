# -*- coding: utf-8 -*-
import os
import json

from azureml.core import Model, Environment, Webservice

from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

import dotenv

from ..utils import *
from ..models import *


def model_deploy(
        ws,
        model_deploy_path,
        show_output=False
    ):
    """Lance le déploiement du modèle sur ACI.
    
    Parameters
    ----------
        ws : Azure Workspace
            Workspace de Azure ML.
        model_deploy_path : str
            Dossier contenant les scripts de déploiement du modèle.
        show_output : bool
            Afficher les logs de sortie.

    Returns
    ----------
        Azure Webservice
            Service web qui fait tourner le script d'inférence du modèle déployé.
    """
    
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
        deployment_config=aciconfig,
        overwrite=True
    )
    
    # On attend la fin du déploiement
    model_aci.wait_for_deployment(show_output=show_output)
        
    # On enregistre l'URL de l'API dans un fichier de variables d'environnement
    dotenv.set_key(os.path.join(model_deploy_path, ".env"), "MODEL_ACI_URL", model_aci.scoring_uri)
        
    return model_aci


# +
if __name__ == "__main__":
    # On charge les variables d'environnemnt
    azure_credentials = json.loads(os.getenv("AZURE_CREDENTIALS"))
    azure_workspace = json.loads(os.getenv("AZURE_WORKSPACE"))
    
    # On charge l’espace de travail Azure Machine Learning existant
    ws = get_ws(azure_credentials, azure_workspace)
    
#     # On supprime le point de terminaison existe déja
#     model_aci = Webservice(ws, "p9-recommender-aci")
#     model_aci.delete()
    
    # On déploie le modèle
    model_aci = model_deploy(
        ws,
        "P9_02_scripts/model_content_based_deploy_aci",
        show_output=True
    )
    
    # On affiche l'URL de l'API
    print(model_aci.scoring_uri)
