# -*- coding: utf-8 -*-
import os
import logging
import json
from datetime import datetime
import requests

from dotenv import load_dotenv

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    # On charge les variables de l'application
    load_dotenv()
    model_aci_url = os.getenv("MODEL_ACI_URL")
    
    logging.info(f"Python HTTP trigger function processed a request at : {model_aci_url}")

    # On récupère l'id de l'utilisateur
    user_id = req.params.get("userId")
    if not user_id:
        try:
            data = req.get_json()
            user_id = data.get("userId")
        except ValueError:
            return func.HttpResponse(
                "Il manque le paramètre 'userId'",
                status_code=200
            )

    # On crée la donnée d'entrée du modèle déployé
    data = {
        "user_id": str(user_id),
        "session_start_dt": datetime(2017, 10, 16, 12).isoformat(),
        "top_n": 5
    }

    # On appelle le service ACI pour effectuer la prédiction des recommendations  
    r = requests.post(model_aci_url, json=data)

    # On vérifie la réponse
    if not r.ok:
        return func.HttpResponse(
             f"Erreur du service ACI {r.status_code}",
             status_code=200
        )

    article_ids = r.json().get("article_ids", [0] * 5)
    
    logging.info(f"Article ids for user {user_id} : {article_ids}")

    # On spécifie que la réponse sera en JSON
    func.HttpResponse.mimetype = "application/json"
    func.HttpResponse.charset = "utf-8"
    
    return func.HttpResponse(json.dumps(article_ids))
