# -*- coding: utf-8 -*-
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException


def get_compute_target(ws, name, location, priority, vm_size, max_nodes=1):
    """"""
    try:
        # On charge le cluster de calcul si il existe
        compute_target = ComputeTarget(workspace=ws, name=name)
    except ComputeTargetException:
        # Si le cluster n'existe pas, on on en crée un nouveau
        config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_NC6",
            location=location,
            vm_priority="lowpriority",
            min_nodes=0,
            max_nodes=1
        )

        compute_target = ComputeTarget.create(
            workspace=ws,
            name=name,
            provisioning_configuration=config
        )
        
        compute_target.wait_for_completion(
            show_output=True,
            min_node_count=None,
            timeout_in_minutes=20
        )
        
    return compute_target
