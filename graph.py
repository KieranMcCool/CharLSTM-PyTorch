#!/usr/bin/env python3
import matplotlib.pyplot as plt

vals = [float(s) for s in open('loss.txt').read().split('\n') if s != '']

plt.plot(range(len(vals)), vals)
plt.show()
