import numpy as np
import matplotlib.pyplot as plt
a=26.82
b=26.4
c=61.17
d=61.55
means   = [a,b,c,d]         # Mean Data 
stds    = [4.59,4.39,4.37,4.38]            # Standard deviation Data
peakval = ['26.82','26.4','61.17','61.55'] # String array of means

ind = np.arange(len(means))
width = 0.35
colours = ['red','blue','green','yellow']

plt.figure()
plt.title('Average Age')
for i in range(len(means)):
    plt.bar(ind[i],means[i],width,color=colours[i],align='center',yerr=stds[i],ecolor='k')
plt.ylabel('Age (years)')
plt.xticks(ind,('Young Male','Young Female','Elderly Male','Elderly Female'))

def autolabel(bars,peakval):
    for ii,bar in enumerate(bars):
        height = bars[ii]
        plt.text(ind[ii], height-5, '%s'% (peakval[ii]), ha='center', va='bottom')
autolabel(means,peakval)
plt.show()
