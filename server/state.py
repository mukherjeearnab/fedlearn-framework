'''
Module for Global State
'''


class GlobalState:
    '''
    Global State Class
    '''

    def __init__(self):
        self.clients = dict()
        self.jobs = dict()

        self.instance("Class INIT")

    def instance(self, caller: str):
        print(f"Global State from {caller} at instance {self}")
        self.print_state()

    def print_state(self):
        print(f"Client State: {self.clients}")
        print(f"Job State: {self.jobs}")
        print(f"Client State: {hex(id(self.clients))}")
        print(f"Job State: {hex(id(self.jobs))}")


# Global State Object
global_state = GlobalState()
