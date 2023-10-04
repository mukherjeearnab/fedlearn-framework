'''
Application Logic for Client Management
'''
from helpers.kvstore import kv_get, kv_set
from helpers.semaphore import Semaphore
from helpers.logging import logger
from apps.job.exec_handler import JobExecHandler

CLIENT_STAGE = {
    0: 'Client Online',
    1: 'Client Ready With Jobsheet',
    2: 'Client Ready With Dataset',
    3: 'Client Busy In Training',
    4: 'Client Waiting For Params',
    5: 'Client Terminated'
}


class JobParamHandler:
    '''
    JobParamHandler Class
    '''

    def __init__(self, project_name: str, load_from_db=False):
        '''
        constructor
        '''

        self.modification_lock = Semaphore()

        self.project_name = project_name

        self.exec_handler = JobExecHandler(self.project_name, {}, {}, {}, True)

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

        if load_from_db:
            self._read_state()
        else:
            self._update_state()

    def _read_state(self):
        payload = kv_get(f'{self.project_name}-execlogic')

        if payload is None:
            logger.error(
                f'No such FedLearn Project named [{self.project_name}] exists!')
        else:
            self.project_name = payload['project_name']
            self.exec_params = payload['exec_params']

    def _update_state(self):
        data = {
            'project_name': self.project_name,
            'exec_params': self.exec_params
        }
        kv_set(f'{self.project_name}-execlogic', data)

    def get_state(self):
        '''
        Get All State Variables of the Job Instance
        '''
        payload = kv_get(f'{self.project_name}-execlogic')

        return payload

    def add_client(self, client_id: str) -> bool:
        '''
        Adds a Client to the list of clients for the current job, only if job_status.process_phase is 0.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.exec_handler.get_job_status('process_phase') == 0:
            self.exec_params['client_info'].append({
                'client_id': client_id,
                'status': 0
            })

            # method suffixed with update state and lock release
            self._update_state()
        else:
            # log output and set execution status to False
            logger.warning(
                f'''Cannot ADD Client! 
                job_status.process_phase is {self.exec_handler.get_job_status("process_phase")}.''')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def update_client_status(self, client_id: str, client_status: int) -> bool:
        '''
        Updates the status of a client, based on their client_id and 
        if all clients have the same status, the global status, i.e., job_status.client_stage is set as the status of the clients
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        all_client_status = set()

        for i in range(len(self.exec_params['client_info'])):
            # find the client and update their status
            if self.exec_params['client_info'][i]['client_id'] == client_id:
                self.exec_params['client_info'][i]['status'] = client_status

            # collect the status of all the clients
            all_client_status.add(
                self.exec_params['client_info'][i]['status'])

        logger.info(
            f"Client [{client_id}] is at stage [{CLIENT_STAGE[client_status]}].")

        if len(all_client_status) == 1:
            self.exec_handler.set_job_status(
                'client_stage', list(all_client_status)[0])
            logger.info(
                f"All clients are at Stage: [{CLIENT_STAGE[self.exec_handler.get_job_status('client_stage')]}]")

            # if all clients is waiting for parameters, update process phase to 2
            # if list(all_client_status)[0] == 4:
            #     self.job_status['process_phase'] = 2
            #     logger.info(
            #         'All clients params are submitted, starting Federated Aggregation.')

        # method suffixed with update state and lock release
        self._update_state()
        self.modification_lock.release()
        return exec_status

    def set_central_model_params(self, params: dict) -> bool:
        '''
        Set or Update the Central Mode Parameters, for initial time, or aggregated update time.
        Only set, if Provess Phase is 0 or 2, i.e., TrainingNotStarted or InCentralAggregation.

        Remember to call allow_start_training() to update the Process Phase to 1,
        to signal Clients to download params and start training.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        # HERER
        if self.exec_handler.get_job_status('process_phase') == 0 or self.exec_handler.get_job_status('process_phase') == 2:
            self.exec_params['central_model_param'] = params

            # empty out client params
            self.exec_params['client_model_params'] = []
            # increment global round
            self.exec_handler.set_job_status('global_round', 1)  # HEREW

            logger.info(
                'Central Model Parameters are Set. Waiting for Process Phase to be in [1] Local Training.')

            # method suffixed with update state and lock release
            self._update_state()
        else:
            logger.warning(
                f'Central model parameters NOT SET! job_status.process_phase is {self.exec_handler.get_job_status("process_phase")}.')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def append_client_params(self, client_id: str, client_params: dict) -> bool:
        '''
        Append Trained Model Params from Clients to the exec_params.client_model_params[].
        Only works if job_status.client_stage=3 and job_status.process_phase=1.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        # HERER
        if self.exec_handler.get_job_status('process_phase') == 1 and self.exec_handler.get_job_status('client_stage') == 3:
            # add client submitted parameters
            self.exec_params['client_model_params'].append({'client_id': client_id,
                                                            'client_params': client_params})

            logger.info(
                f"[{client_id}] submitted params. Total Params: {len(self.exec_params['client_model_params'])}/{self.exec_handler.get_num_clients()}")

            # # update client status to 4, ClientWaitingForParams
            # self.update_client_status(client_id, client_status=4)

            # if all the client's parameters are submitted, set process_phase to 2, i.e., InCentralAggregation
            if len(self.exec_params['client_model_params']) == self.exec_handler.get_num_clients():
                self.exec_handler.set_job_status('process_phase', 2)  # HEREW
                logger.info(
                    'All clients params are submitted, starting Federated Aggregation.')

            # method suffixed with update state and lock release
            self._update_state()
            print("StateUPdatedforAppendClientParams")
        else:
            logger.warning(
                f'Cannot APPEND client model params! job_status.process_phase is {self.exec_handler.get_job_status("process_phase")}, job_status.client_stage is {self.exec_handler.get_job_status("client_stage")}.')  # HERER
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status
