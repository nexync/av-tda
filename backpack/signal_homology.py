import numpy as np
import matplotlib.pyplot as plt
def calculateSignalHomology(y):
    '''
    :param y: numpy array of values 
    :return: birth and death arrays of y
    '''
    x = np.arange(len(y))
    sy = np.sort(y)
    cc = {}
    birth = []
    death = []
    for s in sy:
        ind = np.where(y == s)[0]
        for i in ind:
            if i-1 not in cc and i+1 not in cc:
                cc[i] = s
            else:
                if i-1 not in cc:
                    cc[i] = cc[i+1]
                    birth.append(s)
                    death.append(s)
                elif i+1 not in cc:
                    cc[i] = cc[i-1]
                    birth.append(s)
                    death.append(s)
                else:
                    cc[i] = min(cc[i-1],cc[i+1])
                    if cc[i-1] <= cc[i+1]:
                        birth.append(cc[i+1])
                        death.append(s)
                        j = i+1
                        while j in cc:
                            cc[j] = cc[i-1]
                            j += 1
                    if cc[i-1] > cc[i+1]:
                        birth.append(cc[i-1])
                        death.append(s)
                        j = i-1
                        while j in cc:
                            cc[j] = cc[i+1]
                            j -= 1
    birth.append(sy[0])
    death.append(sy[-1])
    return birth, death

def plotSignalHomology(b,d):
    '''
    param b: birth values from signal homology calculation
    param d: death values from signal homology calculation
    return: none, plots persistence diagram of signal homology
    '''
    fig = plt.figure()
    ax = plt.axes()
    plt.xlim(0, max(d) + 1)
    plt.ylim(0, max(d) + 1)
    ax.scatter(b,d)
    ax.plot(np.arange(max(d)+2),np.arange(max(d)+2))
    plt.show()