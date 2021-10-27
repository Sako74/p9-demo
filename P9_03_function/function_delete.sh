#!/bin/bash
# -*- coding: utf-8 -*-
# On importe les configuration
source function_config.txt

# On supprime le groupe de ressource de l'azure function
az group delete --name $ressourceGroup --yes
