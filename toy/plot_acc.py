from matplotlib import pyplot as plt

steps = [100, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000]
# soft_train_1024_dev256 = [3.333, 3.333, 3.356, 3.344, 3.39, 3.851, 4.689, 5.708, 6.795]
# soft_train_2048_dev256 = [3.846, 3.846, 3.846, 3.861, 4.141, 5.051, 6.4, 7.911, 9.464]
# soft_train_4096_dev256 = [3.846, 3.947, 3.937, 3.906, 3.883, 4.511, 5.698, 7.052, 8.451]
# soft_train_2048_dev128 = [3.846, 3.846, 3.846, 3.846, 4.115, 5.025, 6.359, 7.862, 9.404]

# plt.plot(steps, soft_train_1024_dev256, label="soft_train_1024_dev256")
# plt.plot(steps, soft_train_2048_dev256, label="soft_train_2048_dev256")
# plt.plot(steps, soft_train_4096_dev256, label="soft_train_4096_dev256")
# plt.plot(steps, soft_train_2048_dev128, label="soft_train_2048_dev128")

# soft_dev_mu_0_3_lra_1e_3 = [2.5, 2.564, 2.577, 2.625, 2.903, 3.593, 4.561, 5.643, 6.749]
# soft_dev_mu_0_3_lra_2e_3 = [3.571, 3.614, 3.623, 3.663, 3.976, 4.894, 6.211, 7.669, 9.188]
# soft_dev_mu_0_5 = [3.846, 3.846, 3.846, 3.861, 4.141, 5.051, 6.4, 7.911, 9.464]
# soft_dev_mu_0_5_noise_0_05 = [3.846, 3.846, 3.846, 3.846, 4.115, 5.017, 6.349, 7.837, 9.39]
# soft_train_noise_0_1 = [3.846, 3.846, 3.846, 3.861, 4.141, 5.051, 6.4, 7.911, 9.464]
# soft_train_noise_0_05 = [3.846, 3.846, 3.846, 3.846, 4.115, 5.017, 6.349, 7.837, 9.39]

# soft_train_sigma_0_1 = [3.846, 3.846, 3.846, 3.861, 4.141, 5.051, 6.4, 7.911, 9.464]
# soft_train_sigma_0_2 = [4.762, 5.263, 4.95, 3.906, 4.193, 5.445, 6.993, 8.651, 10.345]

dim128 = [3.846, 3.846, 3.846, 3.861, 4.141, 5.051, 6.4, 7.911, 9.464]
dim256 = [2.564, 2.459, 2.294, 2.024, 2.789, 4.065, 5.405, 6.757, 8.108]
dim512 = [2.564, 1.935, 1.629, 1.812, 3.175, 4.739, 6.319, 7.899, 9.479]

# plt.plot(steps, soft_train_sigma_0_1, label="soft_train_sigma_0.1")
# plt.plot(steps, soft_train_sigma_0_2, label="soft_train_sigma_0.2")
# plt.plot(steps, soft_dev_mu_0_5_noise_0_05, label="soft_dev_mu_0_5_noise_0_05")
# plt.plot(steps, soft_train_noise_0_1, label="soft_train_noise_0_1")
# plt.plot(steps, soft_train_noise_0_05, label="soft_train_noise_0_05")

plt.plot(steps, dim128, label="dim128")
plt.plot(steps, dim256, label="dim256")
plt.plot(steps, dim512, label="dim512")

plt.legend()

plt.savefig("num.png")