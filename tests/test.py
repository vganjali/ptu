import sys
sys.path.append('../ptu')
from src.ptuparser import ptuparser

print(ptuparser.__builtins__)
ptu = ptuparser.PTU()
trace = ptu.load_file('default_007.ptu')
print(trace)