from matplotlib import pyplot as plt

x = [1.0*i/1000 for i in range(1000)]
y = [ ((1+e)/(1-e)) ** (1.0) for e in x ]

plt.plot(x,y)
plt.show()
