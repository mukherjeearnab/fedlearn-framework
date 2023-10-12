class Globals:
    app_globals = {}

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
