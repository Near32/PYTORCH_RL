import csv
import numpy as np
import matplotlib.pyplot as plt


reader = csv.reader(open("./data.csv","rb"), delimiter=",")

x = list(reader)


result = np.array(x)#.astype("float")
titles = result[0,:]

print(titles)

result = result[1:,:]

for i in range(result.shape[0]) :
	for j in range(result.shape[1]) :
		if result[i,j] == '' :
			result[i,j] = '0.0'

result = np.array(result).astype("float")
		
print(result.shape)
#print(result)

# FIGURE :
fig, ax = plt.subplots(1)
colors = ['b','r','k','g']
#
#

X = list()
mu = list()
sigma = list()
windowSize = 50
maxval = 800
inDB = False

for i in range(result.shape[1]) :
	values = result[:maxval+windowSize,i]
	val = list()
	for j in range(values.shape[0]-windowSize) :
		val.append( np.reshape(values[j:j+windowSize], [1,-1] ) )
	val = np.concatenate( val, axis=0)
	
	if inDB:
		val = np.log(val)/np.log(10)
	
	X.append(val)
	mu.append( val.mean(axis=1) )
	sigma.append(  val.std(axis=1) )
	
	t = np.arange(mu[-1].shape[0])
	
	#ax.plot( t, X[-1][:,0], lw=2, label='{}'.format(titles[i]), color=colors[i%len(colors)], alpha=0.8)
	ax.plot( t, mu[-1], lw=2, label='{}'.format(titles[i]), color=colors[i%len(colors)])
	ax.fill_between( t, mu[-1]+sigma[-1], mu[-1]-sigma[-1], facecolor=colors[i%len(colors)], alpha=0.2)



if inDB :
	ax.set_title('Experiment\'s Average Total Returns in dB',fontsize=28)
	ax.set_ylabel(' Average Total Return (over a window of {} episodes) in dB '.format(windowSize),fontsize=28 )
else :
	ax.set_title('Experiment\'s Average Total Returns',fontsize=28)
	ax.set_ylabel(' Average Total Return (over a window of {} episodes) '.format(windowSize),fontsize=28 )

ax.legend(loc='upper left',fontsize=28)
ax.set_xlabel('Episodes',fontsize=28)
ax.grid()

plt.show()
