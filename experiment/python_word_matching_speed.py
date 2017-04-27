#!/usr/bin/python

# Takes in a dictionary of words
# Verifies that all functions return the same answers
# Generates random hands from the probability of getting tiles from the bunch
# Then prints out how long each function takes to find all matching words
# Generates various hand sizes to see if there's any scaling

import sys
import timeit
import numpy as np
import pickle

# Naive list way of matching wordbank
def f0_list(hand, wordbank):
    results = []
    for w_i in range(len(wordbank)):
        match = True
        for i in range(26):
            if hand[i] < wordbank[w_i][i]:
                match = False
                break
        if match:
            results.append(w_i)
    return results

# A for loop and some numpy
def f1_list(hand, wordbank):
    results = []
    for w_i in range(len(wordbank)):
        if min(list(map(lambda x: x[1] - x[0], zip(wordbank[w_i], hand)))) >= 0:
            results.append(w_i)
    return results


# Naive way using numpy
def f0_np(hand, wordbank):
    results = []
    for w_i in range(len(wordbank)):
        match = True
        for i in range(26):
            if hand[i] < wordbank[w_i,i]:
                match = False
                break
        if match:
            results.append(w_i)
    return results


# A for loop and some numpy
def f1_np(hand, wordbank):
    results = []
    for w_i in range(len(wordbank)):
        if not np.any((hand - wordbank[w_i]) < 0):
            results.append(w_i)
    return results


# A for loop and some numpy
def f2_np(hand, wordbank):
    results = []
    for w_i in range(len(wordbank)):
        if np.min(hand - wordbank[w_i]) >= 0:
            results.append(w_i)
    return results


# Vectorized sum and difference
def f3_np(hand, wordbank):
    return np.where(np.sum((wordbank - hand) > 0, axis=1) == 0)[0]


# vectorized just using any
def f4_np(hand, wordbank):
    return np.where(np.any(wordbank > hand, axis=1) == 0)[0]


# Prepare a 2D list and a 2D np array of letter frequencies
with open(sys.argv[1]) as f:
    words = [x.split()[0] for x in f.readlines()]
wordbank_list = [[0] * 26 for _ in range(len(words))]
wordbank_np =  np.zeros((len(words), 26))
for w_i in range(len(words)):
    for letter in sorted(words[w_i]):
        pos = ord(letter) - 65
        wordbank_list[w_i][pos] += 1
        wordbank_np[w_i][pos] += 1


# Arrays for keeping track of functions and data-specific wordbanks
functions = {'list' : [f0_list, f1_list],
             'numpy': [f0_np, f1_np, f2_np, f3_np, f4_np]}
wordbanks = {'list' : wordbank_list,
             'numpy': wordbank_np}
n_iter = 20
timings = {}
for datatype in functions:
    timings[datatype] = {}
    for hand_size in range(2, 9):
        timings[datatype][hand_size] = np.zeros((n_iter, len(functions[datatype])))

        
# Verify that our functions give the same answers
for datatype in functions:
    for func in functions[datatype]:
        print(datatype, func(wordbanks[datatype][len(wordbank_list) // 2], wordbanks[datatype]))
        
        
# Time each word
imports = 'from __main__ import functions, wordbanks'
for counter in range(n_iter):
    print("\r%.3f%%" % (100 * (counter + 1) / n_iter), end='')

    for hand_size in range(2, 9):

        # Get a specific hand size
        hand = [13,3,3,6,18,3,4,3,12,2,2,5,3,8,11,3,2,9,6,9,6,3,3,2,3,2]
        while sum(hand) > hand_size:
            pos = np.random.randint(sum(hand))
            for i in range(len(hand)):
                pos -= hand[i]
                if pos < 0:
                    hand[i] -= 1
                    break
        hand = str(hand)
        
        # For this hand go wild
        for datatype in functions:
            for f_i in range(len(functions[datatype])):
                cmd = 'functions["%s"][%i](%s, wordbanks["%s"])' % (datatype, f_i, hand, datatype)
                timings[datatype][hand_size][counter, f_i] += timeit.timeit(cmd, imports, number=8)
print()


# Save words and timings
with open('word_matching_timings_%s.pkl' % sys.argv[1], 'wb') as f:
    pickle.dump((words, wordbanks, timings), f)

    
# Show Results
print("Data\t#Hand\tF0\tF1\tF2\tF3\tF4")
for datatype in functions:
    for hand_size in range(2, 9):
        print("%s\t%i" % (datatype, hand_size), end='')
        means = np.mean(timings[datatype][hand_size], axis=0)
        for f_i in range(len(functions[datatype])):
            print("\t%7.5f" % (means[f_i]), end='')
        print()
