import math
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# prints formatted price
def formatPrice(n):
	return ("-%" if n < 0 else " %") + "{0:.3f}".format(abs(n))

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


# We don't want random-seeding for reproducibilityy! We _want_ two runs to give different results, because we only
# trust the hyper combo which consistently gives positive results.
def seed( seed=None):
	#np.random.seed(7)
    pass


def plot_histogram(x, bins, title, xlabel, ylabel, xmin=None, xmax=None):
	plt.clf()
	plt.hist(x, bins=bins)
	if xmin != None:
		plt.xlim(xmin, xmax)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig('files/output/' + title + '.png')


def plot_barchart(list, file="BT", title='BT', ylabel="Price", xlabel="Date", colors='green'):
	l = len(list)
	x = range(l)
	# font = {'family': 'serif',
	# 		'color':  'black',
	# 		'weight': 'normal',
	# 		'size': 8,
	# 		}
	myarray = np.asarray(list)
	colors = colors  # 'green'#np.array([(1,0,0)]*l)
	# colors[myarray > 0.0] = (0,0,1)
	plt.clf()
	plt.bar(x, myarray, color=colors)
	#plt.text(0, 0,text,  fontdict=font)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig('files/output/' + file + '.png')


def record_run_time(func):
	"""
	Helper decorator that records the runtime of a given function
    """
	def wrapper(*args, **kwargs):
		print("Current time is: %s" % datetime.now().strftime('%H:%M:%S'))  # episodes=2 +features=252 takes 6 minutes
		start_time = datetime.now()

		# Run the actual function
		func(*args, **kwargs)

		now = datetime.now()
		diff = now - start_time
		minutes = (diff.seconds // 60) % 60
		output = """
Current time is: %s 
Runtime: %s (%d minutes)
Finished run.
        """ % (now.strftime('%H:%M:%S'), diff, minutes)
		print(output)

	return wrapper


