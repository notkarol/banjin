#!/usr/bin/env python3

import sys
import numpy as np

# Number of tiles
num_tiles = int(sys.argv[1])

# Dictionaries to exampne
filenames = sys.argv[2:]
words = {}
for filename in filenames:
    words[filename] = []
    with open(filename) as f:
        for line in f:
            words[filename].append(line.split()[0])

# Prepare frequencies, with the second column containing the frequency of the standard bunch
freqs = np.zeros((len(filenames) + 1, 26), dtype=np.double)
freqs[-1, :] = [13, 3, 3, 6,18, 3, 4, 3,12, 2, 2, 5, 3, 8,11, 3, 2, 9, 6, 9, 6, 3, 3, 2, 3, 2]

# For each file, each word, and each letter, add it to the frquency
for i, filename in enumerate(filenames):
    for word in words[filename]:
        for letter in word:
            freqs[i, ord(letter) - ord('A')] += 1
    freqs[i, :] *= num_tiles / np.sum(freqs[i])

print("Absolute")    
print("Letter\t" + "\t".join(list(map(lambda x: x.split('.')[0], filenames))) + "\tBunch")
for i in range(26):
    print("%c\t%s" % (chr(i + ord('A')), '\t'.join(["%5.2f" % x for x in freqs[:, i]])))

print("Relative")
print("Letter\t" + "\t".join(list(map(lambda x: x.split('.')[0], filenames))) + "\tBunch")
for i in range(26):
    print("%c\t%s" % (chr(i + ord('A')), '\t'.join(["%5.2f" % (x / freqs[-1, i]) for x in freqs[:, i]])))
    
