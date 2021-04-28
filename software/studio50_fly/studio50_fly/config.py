import os
import json

class Config:

    DEFAULT_PATH = os.path.join(os.environ['HOME'], '.config', 'studio50_fly') 
    DEFAULT_FILENAME = 'studio50_fly.json'

    def __init__(self,filename=None): 
        if filename is None:
            self.path = self.DEFAULT_PATH
            self.filename = self.DEFAULT_FILENAME
        else:
            self.path, self.filename = os.path.split(filename)
        self.load(filename)

    def load(self,filename=None):
        if filename is None:
            filename = os.path.join(self.path, self.filename)
        with open(filename, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self,key):
        return self.data[key]

    def __str__(self):
        return str(self.data)



        
