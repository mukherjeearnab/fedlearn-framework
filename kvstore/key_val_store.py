'''
Key Value Store Class
'''


class KeyValueStore:
    def __init__(self):
        self.table = dict()

    def get(self, key: str) -> str:
        '''
        Get Value
        '''
        return self.table[key]

    def set(self, key: str, value: str) -> None:
        '''
        Set Value
        '''
        self.table[key] = value

    def check(self, key: str) -> bool:
        '''
        Check if Key has been Set
        '''
        return key in self.table.keys()
            