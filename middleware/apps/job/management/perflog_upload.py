from helpers.perflog import add_params, add_record, save_logs
from apps.job.api import get_job


def upload_perflog_metrics(middleware_id: str, job_name: str, metrics: dict, params: str, time_delta: float):
    '''
    Upload Round Metrics and Infor to Perflog server.
    '''
    state = get_job(job_name)
    add_params(job_name, f"{state['job_status']['global_round']}",
               params)
    add_record(middleware_id, job_name, metrics,
               f"{state['job_status']['global_round']}", time_delta)
    save_logs(job_name)
