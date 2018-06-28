import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
# import twitter
import time
import connect

style.use('ggplot')

fig = plt.figure()
# plt.xlabel("Hello")
ax1 = fig.add_subplot(1,1,1)
X =""
# def Xvalue(tweet):
#     X = tweet

def animate(i):
    pullData = open("twitter-out.txt","r").read()
    lines = pullData.split("\n")

    xar = []
    yar = []
    x=0
    y=0
    for l in lines[-100:]:
        x += 1
        if "1" in l:
            y += 1
        elif "0" in l:
            y -= 1

        xar.append(x)
        yar.append(y)
    ax1.clear()
    ax1.plot(xar,yar)
    ax1.set_title("Sentiment Graph")
    plt.xlabel(connect.topic)


def plotg():

    ani = animation.FuncAnimation(fig,animate,interval=10)

    plt.show()
    time.sleep(1)

if __name__=="__main__":
    plotg()


