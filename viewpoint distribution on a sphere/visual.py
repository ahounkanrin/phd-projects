import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


df = pd.read_csv("test500.csv", sep=",")
x = df["x"]
y = df["y"]
z = df["z"]
fig = plt.figure(figsize=(12, 12))
fig.suptitle("Viewpoints distribution on a sphere")
ax = fig.add_subplot(221, projection="3d")
ax.scatter(x, y, z, alpha=0.1)
ax.set_title("500 samples")
#plt.show()

df = pd.read_csv("test1000.csv", sep=",")
x = df["x"]
y = df["y"]
z = df["z"]
#fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(222, projection="3d")
ax.scatter(x, y, z, alpha=0.1)
ax.set_title("1000 samples")
#plt.show()


df = pd.read_csv("test2000.csv", sep=",")
x = df["x"]
y = df["y"]
z = df["z"]
#fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(223, projection="3d")
ax.scatter(x, y, z, alpha=0.1)
ax.set_title("2000 samples")
#plt.show()

df = pd.read_csv("test5000.csv", sep=",")
x = df["x"]
y = df["y"]
z = df["z"]
#fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(224, projection="3d")
ax.scatter(x, y, z, alpha=0.1)
ax.set_title("5000 samples")
plt.show()

