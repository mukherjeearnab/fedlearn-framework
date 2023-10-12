from helpers.file import create_dir_struct
from helpers.http import download_file


def download_upstream_dataset(job_id: str, middleware_id: str, server_url: str):
    '''
    Downloads Dataset Chunk assiened for the middleware from the Upstream Server
    '''
    file_name = f'{middleware_id}.tuple'
    dataset_path = f'./datasets/{job_id}'
    create_dir_struct(dataset_path)
    download_file(f'{server_url}/job_manager/download_dataset?job_id={job_id}&client_id={middleware_id}',
                  f'{dataset_path}/{file_name}')
