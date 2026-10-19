import os, sys
with open(sys.argv[1],'wb') as f:
    for _ in range(200): f.write(os.urandom(196608))
