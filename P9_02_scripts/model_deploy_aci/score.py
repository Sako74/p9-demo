# -*- coding: utf-8 -*-
import json
from datetime import datetime
# import logging

from training import *


# +
# logging.basicConfig(level=logging.DEBUG)

# +
def init():
    global model
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)
    
#     MODEL_PATH = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    
#     # On charge le modèle
#     model = joblib.load(os.path.join(MODEL_PATH, "content_based_recommender.joblib"))


# -

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

    # On met à jour les données
    data.update({"article_ids": article_ids})

    # On renvoie un dictionnaire au même format que les données
    # en entrée, mais complété par les prédictions du modèle.
    return data
