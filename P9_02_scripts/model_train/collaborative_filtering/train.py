# -*- coding: utf-8 -*-
import time

from azureml.core import Run, Dataset

from training import *

# On crée le dossier de sortie s'il n'existe pas
OUTPUT_PATH = "outputs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

parser = argparse.ArgumentParser()

# Id des datasets
parser.add_argument("--train_user_article_ratings_id", type=str)
parser.add_argument("--valid_user_article_ratings_id", type=str)
parser.add_argument("--article_profiles_id", type=str)

# Hyperparamètres
parser.add_argument("--rating_col", type=str)
parser.add_argument("--n_factors", type=int)
parser.add_argument("--n_epochs", type=int)
parser.add_argument("--lr_all", type=float)
parser.add_argument("--reg_all", type=float)

args = parser.parse_args()

print("On récupère le contexte et le workspace.")

run = Run.get_context()
ws = run.experiment.workspace

print("On charge les datasets dans des DataFrame.")

train_user_article_ratings = Dataset.get_by_id(ws, id=args.train_user_article_ratings_id).to_pandas_dataframe()
valid_user_article_ratings = Dataset.get_by_id(ws, id=args.valid_user_article_ratings_id).to_pandas_dataframe()
article_profiles = Dataset.get_by_id(ws, id=args.article_profiles_id).to_pandas_dataframe()

print("On filtre les colonnes pour obtenir le format : user_id, article_id, rating_id.")

train_ratings = train_user_article_ratings[["user_id", "article_id", args.rating_col]]
train_ratings = train_ratings.rename(columns={args.rating_col: "rating"})

valid_ratings = valid_user_article_ratings[["user_id", "article_id", args.rating_col]]
valid_ratings = valid_ratings.rename(columns={args.rating_col: "rating"})

print("On crée le modèle.")

# +
kwargs = {   
    "n_factors": args.n_factors,
    "n_epochs": args.n_epochs,
    "lr_all": args.lr_all,
    "reg_all": args.reg_all,
}

model = CollaborativeRecommender(
    article_profiles,
    train_ratings,
    **kwargs
)
# -

print("On entraine le modèle.")

start_time = time.time()

model.fit()

stop_time = time.time()
training_time = stop_time - start_time

run.log("training_time_s", training_time)

print(f"Temps total de l'entrainement: {training_time:0.1f} s")

print("On enregistre le modèle.")

joblib.dump(model, OUTPUT_PATH + "model.joblib")

print("On évalue le modèle.")

start_time = time.time()

res = get_precision_recall_n_score(
    model,
    "CollaborativeRecommender",
    valid_ratings,
    article_profiles,
    top_n=5
)

stop_time = time.time()
evaluation_time = stop_time - start_time

run.log("evaluation_time_s", evaluation_time)

print(f"Temps total de l'évaluation: {evaluation_time:0.1f} s")

print("On enregistre les résultats de l'évaluation.")

run.log("precision@5", res.iloc[0]["precision@5"])
run.log("recall@5", res.iloc[0]["recall@5"])

res.to_parquet(OUTPUT_PATH + "res.parquet", index=False)
