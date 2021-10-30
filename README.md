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

## Création de la ressource Azure Machine Learning

## Ajout du secret AZURE_WORKSPACE

## Ajout du secret AZURE_CREDENTIALS

## Création de l'Azure function

## Ajout du secret AZURE_FUNCTIONAPP_PUBLISH_PROFILE

# Organisation du projet

Les notebooks ont été développés via l'interface web du Studio de Microsoft Azure Machine Learning.

Toutes les dépendances sont listées dans les fichiers `*.yml`.

Tous les scripts sont disponibles dans le dossier `/scripts`.

# Utilisation des notebooks

Pour utiliser les notebooks, importer ce projet dans le Studio de Microsoft Azure Machine Learning.

- Lancer une instance de calcul.
- Ouvrir un terminal et aller dans le répertoire du projet :
- Créer un environnement virtuel conda et installer les dépendances :
```
./conda_create_env.sh
```

# Test de l'application mobile
