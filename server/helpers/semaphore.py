from threading import Semaphore as Original


class Semaphore(Original):
    def wait(self):
        '''
        New Function wait
        '''
        while self._value == 0:
            pass

    