import sys
sys.path.append('../ptu')
from src.ptuparser import ptuparser

ptu = ptuparser.ptu()
ptu.processHT2('./tests/default_007.ptu',binsize=1e-5)