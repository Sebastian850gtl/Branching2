
import matplotlib.pyplot as plt
import numpy as np

# Parameters
save_path = ''
fig_path = ''
fig_name = "Figure5_1"

D1 = 1
D2 = 0.5
r1 = 0.01
#=======================

radius_range = np.load(save_path + "vaues_of_r2.npy")
meanTs = np.load(save_path + "means.npy")
Terr_95 = np.load(save_path + "Terr_95.npy")

f = lambda x : (-np.log(r1 + x) + np.log(2) )* 1/(D1 + D2)
radius_linspace = np.linspace(radius_range[0],radius_range[-1],200)

intercept = np.mean(f(radius_range) - meanTs)
print(intercept) 
g = lambda x : f(x) - intercept

plt.figure(dpi = 200)

plt.plot(radius_linspace,g(radius_linspace),label = r'$r_2 \mapsto \dfrac{-\log\left(%.4f + r_2\right) + \log(2) }{%.1f + %.1f} + %.2f$'%(r1,D1,D2,intercept),
            color = 'darkorange')
plt.scatter(radius_range,meanTs,label = r'$\hat{T}$',color = 'blue')
plt.errorbar(radius_range,meanTs, yerr=Terr_95, fmt='o', color='blue', capsize=5)
plt.legend()
plt.ylabel("Time")
plt.xlabel(r"$r_2$")
plt.savefig(fig_path+fig_name+'.png')
plt.show()