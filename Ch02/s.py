import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataSet[:,1],datingDataSet[:,2])
	       
plt.show()

