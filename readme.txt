
------12/20/2020 stange tensorflow out of memory problem ---
this occurs on homepc win10 with a titan gpu. even for a small test like
testtf.py, it try to get all gpu memory (19 GB) and error out of memory.

this is temprary solved by using the gpu memeory limit option in tf. see testtf.py
and testmodels.py

