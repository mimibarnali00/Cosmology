import numpy as np

a = np.linspace(1,2,10)
#b = np.linspace(2,2.5,10)
#c = np.linspace(3,4,10)

#file = open("sample.txt", "w")
#content1 = str(a)
#content2 = str(b)
#content3 = str(c)
#L = ["a \n", content1, "\n","b \n",content2,"\n", "c \n",content3,"\n"]

#file.writelines(L)
#file.close()

#a = ("a",np.linspace(1,2,10))
b = np.linspace(2,2.5,10)
c = np.linspace(3,4,10)

#file = open("sample.txt", "w")
#content1 = str(a)
#content2 = str(b)
#content3 = str(c)
#L = [content1, "\n","b \n",content2,"\n", "c \n",content3,"\n"]

#file.writelines(L)
#file.close()

#f = open("sample.txt", "r")
#print(f.read())

np.savetxt('sample1.txt', np.array([a, b, c]).T, delimiter='\t', fmt="%s",header='a b c')

