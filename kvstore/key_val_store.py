'''
Key Value Store Class
'''


class KeyValueStore:
    '''
    Key Val Store Class
    '''
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

    def delete(self, key: str) -> None:
        '''
        Delete Value
        '''
        del self.table[key]

    def check(self, key: str) -> bool:
        '''
        Check if Key has been Set
        '''
        return key in self.table
            