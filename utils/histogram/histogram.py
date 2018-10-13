import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import time 

def histogram(ax,x,y) :
	ax.clear()
	numBins = 100
	ax.hist(x,numBins,color='green',alpha=0.4)
	ax.hist(y,numBins,color='blue',alpha=0.4)
	plt.pause(0.1)


class HistogramDebug(object) :
	def __init__(self,numBins=10,alpha=0.4,colors=['black','red','green','blue']):#,'yellow']) :
		self.numBins = numBins
		self.alpha = alpha
		self. colors = colors
		self.nbrSimultaneousDebugging = len(self.colors)

		self.data = []
		for i in range(self.nbrSimultaneousDebugging+1) :
			self.data.append(np.random.normal(0,1,100))

		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111)
		self.ll = None
		self.lr = None

		self.update()

	def update(self) :
		plt.ion()
		self.ax.clear()
		colidx = 0

		for i in range(self.nbrSimultaneousDebugging) :
			self.ax.hist( self.data[-(i+1)], self.numBins, color=self.colors[colidx%self.nbrSimultaneousDebugging], alpha=self.alpha)
			colidx += 1
		
		self.ax.set_xlim(left=self.ll, right=self.lr)
		
		plt.pause(0.1)

	def append(self,x) :
		self.data.append(x)
		self.update()

	def setXlimit(self,ll,lr):
		self.lr = lr
		self.ll = ll

'''
class Histogram3DDebug(object) :
	def __init__(self,numBins=10,alpha=0.4,colors=['black','red','green','blue']):#,'yellow']) :
		self.numBins = numBins
		self.alpha = alpha
		self. colors = colors
		self.nbrSimultaneousDebugging = len(self.colors)

		self.data = []
		for i in range(self.nbrSimultaneousDebugging+1) :
			self.data.append(np.random.normal(0,1,100))

		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111,projection='3d')
		self.ll = None
		self.lr = None

		self.update()

	def update(self) :
		plt.ion()
		self.ax.clear()
		colidx = 0

		
		for i in range(self.nbrSimultaneousDebugging) :
			xs = np.arange(self.numBins)
			ys = self.data[-(i+1)]
			zs = i*10
			cs = [self.colors[colidx%self.nbrSimultaneousDebugging]] * len(xs) 
			self.ax.hist( xs, ys, zs, zdir='y', color=cs, alpha=self.alpha)
			colidx += 1
		
		self.ax.set_xlim(left=self.ll, right=self.lr)
		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Y')
		self.ax.set_zlabel('Z')
		
		plt.pause(0.1)

	def append(self,x) :
		self.data.append(x)
		self.update()

	def setXlimit(self,ll,lr):
		self.lr = lr
		self.ll = ll

'''

def test1() :
	fig = plt.figure()
	ax = fig.add_subplot(111)

	for i in range(100) :
		x = np.random.normal(0,1,1000)
		y = np.random.normal(0.5,2,1000)
		histogram(ax,x,y)
		time.sleep(1)

def test2() :
	hd = HistogramDebug()

	for i in range(100) :
		x = np.random.normal(0,1,1000)
		hd.append(x)
		time.sleep(0.5)
'''
def test3() :
	hd = Histogram3DDebug()

	for i in range(100) :
		x = np.random.normal(0,1,1000)
		hd.append(x)
		time.sleep(0.5)
'''

def test4() :
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
	    xs = np.arange(20)
	    ys = np.random.rand(20)

	    # You can provide either a single color or an array. To demonstrate this,
	    # the first bar of each set will be colored cyan.
	    cs = [c] * len(xs)
	    cs[0] = 'c'
	    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.show()


def test5() :
	import threading
	import time

	def plot_a_graph(f=False):
	    f,a = plt.subplots(1)
	    line = plt.plot(range(10))
	    #plt.show()
	    plt.pause(1)
	    plt.draw()
	    print("plotted graph")    
	    time.sleep(4)

	target = lambda  : plot_a_graph(True)
	testthread = threading.Thread(target=target)

	#plot_a_graph()      # this works fine, displays the graph and waits
	print("that took some time")

	testthread.start() # Thread starts, window is opened but no graph appears
	print("already there")


if __name__ == '__main__' :
	#test1()
	#test2()
	#test3()
	#test4()
	test5()