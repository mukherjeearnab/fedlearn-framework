'''
Key Value Store Class
'''


class KeyValueStore:
    '''
    Key Val Store Class
    '''

    def __init__(self):
        self.user_tables = dict()

    def get(self, key: str, user: str) -> str:
        '''
        Get Value
        '''
        return self.user_tables[user][key]

    def set(self, key: str, value: str, user: str) -> None:
        '''
        Set Value
        '''
        # init table for user if not exist
        if user not in self.user_tables:
            self.user_tables[user] = dict()

        self.user_tables[user][key] = value

    def delete(self, key: str, user: str) -> None:
        '''
        Delete Value
        '''

        # init table for user if not exist
        if user not in self.user_tables:
            self.user_tables[user] = dict()

        del self.user_tables[user][key]

    def check(self, key: str, user: str) -> bool:
        '''
        Check if Key has been Set
        '''
        # init table for user if not exist
        if user not in self.user_tables:
            self.user_tables[user] = dict()

        return key in self.user_tables[user]
