from matplotlib import pyplot as plt

B1 = 1
beta1 = 0.075
c1 = 1 - B1

B2 = 0.5
beta2 = 0.25
c2 = 1 - B2

B3 = 0.2
beta3 = 1.5
c3 = 1 - B3

B4 = 3.2
beta4 = 0.05
c4 = 1.364 - B4
a4 = -10

def f(x, B, beta, c, a=0):
    return B / ((x-a) ** beta) + c


X = [x for x in range(1, 50)]

Y1 = [f(x, B1, beta1, c1) for x in X]
Y2 = [f(x, B2, beta2, c2) for x in X]
Y3 = [f(x, B3, beta3, c3) for x in X]
Y4 = [f(x, B4, beta4, c4, a4) for x in X]

fig = plt.figure(figsize=(4, 3))
ax1 = fig.add_subplot(111)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim(0, 50)
# ax1.set_ylim(0.67, 1)
ax1.axis("off")

l1, = ax1.plot(X, Y1, color="blue", linewidth=2.5)
l3, = ax1.plot(X, Y3, color="blue", linestyle="--", alpha=0.5, linewidth=2.5)
l4, = ax1.plot(X, Y4, color="blue", linestyle="--", alpha=0.5, linewidth=2.5)
l2, = ax1.plot(X, Y2, color="red", linewidth=2.5)

plt.savefig("/home/lidong1/yuxian/sps-toy/toy/trm/tools/toy.pdf", bbox_inches="tight")
