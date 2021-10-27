# -*- coding: utf-8 -*-
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
from datetime import datetime
import logging

from models import *

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


def init():
    global model
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)


def run(data_json):
    """
    La donnée doit respecter le format de l'exemple suivant :
    {
        "user_id": "1234",
        "session_start_dt": "2017-10-16T12:00:00",
        "top_n": 5,
    }
    """

    # On extrait les données
    data = json.loads(data_json)

    user_id = np.int64(data["user_id"])
    session_start_dt = datetime.fromisoformat(data["session_start_dt"])
    top_n = int(data["top_n"])

    # On effectue la prédiction des recommandations
    article_ids = model.recommand(user_id, session_start_dt, top_n=top_n)
    article_ids = [str(i) for i in article_ids]
    
    # On log la prédiction afin de pouvoir la monitorer
    logging.info(f"Article ids for user {user_id} : {article_ids}")

    # On met à jour les données
    data.update({"article_ids": article_ids})

    # On renvoie un dictionnaire au même format que les données
    # en entrée, mais complété par les prédictions du modèle.
    return data
