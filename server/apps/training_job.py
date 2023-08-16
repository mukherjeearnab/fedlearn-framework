'''
Application Logic for Client Management
'''
from helpers.kvstore import kv_get, kv_set
from helpers.semaphore import Semaphore


class TrainingJobManager:
    '''
    Training Job Manager Class
    '''

    def __init__(self, project_name: str, num_clients=1, dataset_dist_file='def_1c.py',
                 dataset_id='mnist_def', train_ratio=0.8, test_ratio=0.2, val_ratio=0.0,
                 model_file='simple_nn', dataset_preprocess_file='mnist_def',
                 training_loop_file='simple_loop', test_file='simple_test',
                 learning_rate=0.01, rounds=10, batch_size=64, local_epochs=1):
        '''
        constructor
        '''

        self.modification_lock = Semaphore()

        self.project_name = project_name

        self.dataset_params = {
            'num_clients': num_clients,
            'dataset_dist_file': dataset_dist_file,
            'dataset_id': dataset_id,
            'ratio': {
                'train': train_ratio,
                'test': test_ratio,
                'validation': val_ratio
            }
        }

        self.model_params = {
            'model_file': model_file,
            'dataset_preprocess_file': dataset_preprocess_file,
            'training_loop_file': training_loop_file,
            'test_file': test_file
        }

        self.train_params = {
            'learning_rate': learning_rate,
            'rounds': rounds,
            'batch_size': batch_size,
            'local_epochs': local_epochs
        }

        self.job_params = {
            'client_stage': 0,
            'download_jobsheet': False,
            'download_dataset': False,
            'process_phase': 0,
            'extra_params': dict()
        }

        self.exec_params = {
            'client_info': [
                # {
                #     'client_id': 'client-0',
                #     'status': 0
                # }
            ],
            'client_model_params': [],  # state_dict()
            'central_model_param': None,  # state_dict(),
        }

        self._update_state()

    def _read_state(self):
        payload = kv_get(self.project_name)

        if payload is None:
            self.__init__()
        else:
            self.project_name = payload['project_name']
            self.dataset_params = payload['dataset_params']
            self.model_params = payload['model_params']
            self.train_params = payload['train_params']
            self.job_params = payload['job_params']
            self.exec_params = payload['exec_params']

    def _update_state(self):
        data = {
            'project_name': self.project_name,
            'dataset_params': self.dataset_params,
            'model_params': self.model_params,
            'train_params': self.train_params,
            'job_params': self.job_params,
            'exec_params': self.exec_params
        }
        kv_set(self.project_name, data)

    def allow_jobsheet_download(self) -> None:
        '''
        Allows to download Jobsheet for clients to prepare themselves.
        Basically modifies the job_params.download_jobsheet from False to True.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()

        # method logic
        self.job_params['download_jobsheet'] = True

        # method suffixed with update state and lock release
        self._update_state()
        self.modification_lock.release()

    def allow_dataset_download(self) -> None:
        '''
        Allows to download dataset for clients to prepare themselves from the data-distributor.
        Basically modifies the job_params.download_dataset from False to True.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()

        # method logic
        self.job_params['download_dataset'] = True

        # method suffixed with update state and lock release
        self._update_state()
        self.modification_lock.release()

    def add_client(self, client_id: str) -> None:
        '''
        Adds a Client to the list of clients for the current job, only if job_params.process_phase is 0.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()

        # method logic
        if self.job_params['process_phase'] == 0:
            self.exec_params['client_info'].append({
                'client_id': client_id,
                'status': 0
            })

            # method suffixed with update state and lock release
            self._update_state()
        self.modification_lock.release()

    def update_client_status(self, client_id: str, client_status: int) -> None:
        '''
        Updates the status of a client, based on their client_id and 
        if all clients have the same status, the global status, i.e., job_params.client_stage is set as the status of the clients
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()

        # method logic
        all_client_status = set()

        for i in range(len(self.exec_params['client_info'])):
            # find the client and update their status
            if self.exec_params['client_info'][i]['client_id'] == client_id:
                self.exec_params['client_info'][i]['status'] = client_status

            # collect the status of all the clients
            all_client_status.add(
                self.exec_params['client_info'][i]['status'])

        if len(all_client_status) == 1:
            self.job_params['client_stage'] = list(all_client_status)[0]

        # method suffixed with update state and lock release
        self._update_state()
        self.modification_lock.release()
