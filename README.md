# Introduction

Ce projet a pour but la réalisation d'un MVP d'un système de recommandation de contenu sous la forme d'une application mobile.

Fonctions de l'application mobile :
- Simulation de l’authentification d’un utilisateur.
- Recommandation de 5 articles.

Fonctions et contraintes du système de recommandation :
- Utilisation des services de Azure.
- Stockage des scripts sur Github.
- Anticiper l’ajout de nouveaux utilisateurs.
- Anticiper l’ajout de nouveaux articles.

<div align="center">
  <img src="./P9_01_notebooks/data/gif/D%C3%A9monstration%20MVP%20mini.gif" alt="Démonstration MVP" style="width:200px;"/>
</div>
<p align="center">Démonstration du MVP</p>

Nous avons aussi mis en place une architecture MLOps qui va nous permettre d'itérer rapidement sur le MVP.

![Architecture MLOps du projet](./P9_01_notebooks/data/img/Architecture%20MLOps.png)
<p align="center">Architecture MLOps</p>

# Création du l'infrastructure

Pour utiliser ce projet, il faut commencer par créer plusieurs ressources sur Microsoft Azure.

## Création de la ressource Azure Machine Learning

Suivez les instructions de ce tutoriel : [Démarrage rapide : créer les ressources d’espace de travail nécessaires pour commencer à utiliser Azure Machine Learning](https://docs.microsoft.com/fr-fr/azure/machine-learning/quickstart-create-resources).

Cloner ce projet sur une instance de calcul.

Ouvrir une invite de commande et aller dans le dossier du projet.

Taper les lignes suivantes afin d'installer l'environnement conda nécessaire à l'exécution des notebooks :
```
conda env create --name p9 --file P9_01_notebooks/conda_env.yml
conda activate p9
python -m ipykernel install --user --name=p9
```

## Ajout du secret AZURE_WORKSPACE

Ajouter le secret AZURE_WORKSPACE dans votre repository Github avec la valeur suivante :
```
{
  "resourceGroup": "<nom du groupe de ressources de azure ml>",
  "workspaceName": "<nom du workspace de azure ml>"
}
```

## Ajout du secret AZURE_CREDENTIALS

Il va falloir donner l'autorisation à Github d'accéder aux resources de Microsoft Azure. Pour cela, nous allons créer une Service Pincipal sur Azure.

Sur votre compte Azure, ouvrez une invite de commande et taper la commande suivante :
```
# Replace {service-principal-name}, {subscription-id} and {resource-group} with your 
# Azure subscription id and resource group name and any name for your service principle
az ad sp create-for-rbac --name {service-principal-name} \
                         --role contributor \
                         --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
                         --sdk-auth
```

Ajouter le secret AZURE_CREDENTIALS dans votre repository Github avec la valeur renvoyée par la commande précédente.

## Création de l'Azure function

Ouvrir une invite de commande et aller dans le dossier du projet.

Taper les commandes suivantes afin de créer les ressources qui vont permettre de déployer l'Azure function :
```
conda activate p9
cd P9_03_function/
./function_create.sh
```

## Ajout du secret AZURE_FUNCTIONAPP_PUBLISH_PROFILE

Ajouter le secret AZURE_FUNCTIONAPP_PUBLISH_PROFILE dans votre repository Github avec la valeur renvoyée à la fin de la commande précédente.

Ce secret est utilisé par la Github action [Azure/functions-action@v1](https://github.com/marketplace/actions/azure-functions-action) qui permet de déployer le code du dossier P9_03_function/recommender.

# Organisation du projet

Les notebooks ont été développés via l'interface web du Studio de Microsoft Azure Machine Learning.

Toutes les dépendances sont listées dans les fichiers `*.yml`.

Tous les scripts sont disponibles dans le dossier `/scripts`.

# Test de l'application mobile
