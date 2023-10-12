from apps.job.api import set_central_model_params


def set_downstream_central_model_params(job_name: str, params: str):
    '''
    Sets Downstream Central Model Parameters
    '''

    set_central_model_params(job_name, params)
