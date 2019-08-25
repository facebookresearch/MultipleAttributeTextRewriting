import sys
import json


assert len(sys.argv) == 4
path = sys.argv[1]
k = sys.argv[2]
t = sys.argv[3]
assert t in ['i', 'f', 's']


with open(path, 'r') as f:
    for line in f:
        parsed = json.loads(line)[k]
        if t == 'i':
            assert type(parsed) is int
            print(parsed)
        if t == 'f':
            assert type(parsed) is float
            print(int(parsed))
        if t == 's':
            assert type(parsed) is str
            print(parsed.replace('\n', ' ').replace('\t', ' ').strip())
