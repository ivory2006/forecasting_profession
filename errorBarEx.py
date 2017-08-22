import matplotlib.pyplot as plt

colors = ["b","g","c","m","y","k","r","g","c","m","y","k",
          "b","g","c","m","y","k","r","g","c","m","y","k"]

xlabels = ['A','B','8','14']

xval = [0, 1, 2, 3]
yval = [0, 1, 4, 9]
yerr = [0.5, 0.4, 0.6, 0.9]

plt.scatter(xval, yval, c=colors, s=50, zorder=3)
plt.errorbar(xval, yval, yerr=yerr, zorder=0, fmt="none",
             marker="none")

plt.savefig("scatter_error.png", dpi=300)
plt.show()
