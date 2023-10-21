import pickle
import numpy as np

file = open("out1.bin", "rb")

bs = pickle.load(file)

print(bs)