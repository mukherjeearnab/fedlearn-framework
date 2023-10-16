import time
import pickle


class Globals:
    app_globals = {}

    state_instance = int(time.time())

    def load(self):
        '''
        Load the App Globals Dict from File
        '''

        self.app_globals = pickle.load(
            open(f'./temp/global_state-{self.state_instance}.temp', 'rb'))

    def save(self):
        '''
        Save The Global State to File
        '''
        pickle.dump(self.app_globals, open(
            f'./temp/global_state-{self.state_instance}.temp', 'wb'))

    def get(self, key: str):
        '''
        Get the value of a given key
        '''
        if key in self.app_globals:
            return self.app_globals[key]
        else:
            return None

    def set(self, key: str, value):
        '''
        Set a key/value pair
        '''
        self.app_globals[key] = value


app_globals = Globals()
