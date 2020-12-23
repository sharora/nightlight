from multiprocessing.connection import Client
import random

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

while(True):
    client.send([random.randrange(11)-5, random.randrange(11)-5])

