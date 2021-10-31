# -*- coding: utf-8 -*-
import os
import subprocess


# On récupère la liste des notebooks à convertir (avec les sorties)
file_names = os.listdir()
notebooks = [f.replace(".ipynb", "") for f in file_names if f.endswith(".ipynb") and not f.endswith("no_out.ipynb")]

for n in notebooks:
    # On crée un version nettoyée du notebook (sans les sorties)
    args = f"jupyter nbconvert --clear-output --to notebook --output {n}_no_out {n}.ipynb".split()
    subprocess.run(args)
