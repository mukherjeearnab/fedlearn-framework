'''
Semaphore Wrapper Module
'''
from threading import Semaphore as Original


class Semaphore(Original):
    '''
    Semaphore Wrapper Class to add wait functionality
    '''
    def wait(self):
        '''
        New Function wait
        '''
        while self._value == 0:
            pass
    