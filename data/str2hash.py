#!/home/subramas/anaconda3/bin/python
import hashlib
import sys

if __name__ == '__main__':
    for line in sys.stdin:
        hash_object = hashlib.sha256(line.strip().replace(' ', '').encode())
        hex_dig = hash_object.hexdigest()
        print(hex_dig)
