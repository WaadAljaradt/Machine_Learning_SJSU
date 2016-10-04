import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random
import time

n = 100
update = 0

#generate random data (x1,x2) x0=1 use x1+0.5x2 +1 >0 for y(x)
def generateData(pts):
    i=0
    for i in range(0,n):
        datapoint = (1, random.randint(-100,100), random.randint(-100,100))
    	if((datapoint[1]+(0.5*datapoint[2])+1) > 0):
           	pts.append([datapoint, 1])
        else:
        	pts.append([datapoint, -1])
        	
#h(x) = wT dot x       	
def hypothesis (x,w):
	return int(np.sign(w.T.dot(x)))        

#check if x is miss_classified by comparing h(x) with f(x) and add it to miss_classified array            
def is_missclassified(w, pts):
    miss_classified = []
    for x, y in pts:
        h =hypothesis(x,w)
        if h!= y:
        	miss_classified.append((x, y))
    if miss_classified :
    	return random.choice(miss_classified)
    return miss_classified #return a randomly chosen x to use for correction of w


#PLA learning iteration
def pla(pts):
	global update 
	w = np.zeros(3) 
	while is_missclassified(w, pts):
		miss_classified = is_missclassified(w, pts)
		x, y = miss_classified
		w += np.dot(y , x)
		update += 1 # times of updating w for pla converge 
	return w
    
#h(x) array
h=[]
 
#data points 
pts = []

#generate random data 
generateData(pts)

#calculate time of convergence 
start_time = time.time()

#get w vector from pla iterations
w = pla(pts)

#print calculation time	
print(time.time() - start_time)

#represent y regions
pos =[]
neg =[]
for x, y in pts:
	if y == 1 :
		pos.append(x)
	else:
		neg.append(x)


print "times of updating w : " , update

#plot results 

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter([v[1] for v in pos], [v[2] for v in pos], c='b', marker="o")
ax1.scatter([v[1] for v in neg], [v[2] for v in neg], c='r', marker="x")
plt.xlabel('x1')
plt.ylabel('x2')
l = np.linspace(-100,100)

#hypothesis seperation line 
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-', label='h(x)')

#f(x) line with green 
ax1.plot(l, -2*l - 2, c='g',label ='f(x)')


plt.legend()
plt.show()