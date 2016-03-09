import  numpy as np

x = np.array([[1,2,6],[3,4,6],[0,0,0]])
y = np.array([[5,6,7],[7,8,9],[0,0,0]])


b=[1,2,3,4,5]

a=np.array([0,1,2])

x[x>2]=-1
m,n=a.shape

print x-a


