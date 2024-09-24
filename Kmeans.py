# Load packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df_penguins = pd.read_csv('penguins_size.csv').dropna()
print(df_penguins)

#g = sns.PairGrid(df_penguins)
#g.map(sns.scatterplot)
#plt.show()

X = np.array(df_penguins.loc[:,['culmen_length_mm',                # Choose your variable names
                       'culmen_depth_mm']]).reshape(-1, 2) # The -1 is 'adjust this parameter to make data fit' So it has 2 columns here
#print(X)

# Determine optimal cluster number with elbow method
wcss = []

for i in range(1, 11): # 1 to 10 clusters
    model = KMeans(n_clusters = i,
                   init = 'k-means++',                 # init what the centroids are. 'k-means' selects based on sampling and speeds up
                   max_iter = 300,                     # Maximum number of iterations
                   n_init = 10,                        # Choose how often algorithm will run with different centroid
                   random_state = 1234)                   # Choose random state for reproducibility
    model.fit(X)
    wcss.append(model.inertia_)

# Show Elbow plot
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')                               # Set plot title
plt.xlabel('Number of clusters')                        # Set x axis name
plt.ylabel('Within Cluster Sum of Squares (WCSS)')      # Set y axis name
plt.show()


kmeans = KMeans(n_clusters = 3,                 # Choose k = 3 clusters from elbow graph
                init = 'k-means++',             # Initialization method for kmeans
                max_iter = 300,                 # Maximum number of iterations
                n_init = 10,                    # Choose how often algorithm will run with different centroid
                random_state = 1234)               # Choose random state for reproducibility

pred_y = kmeans.fit_predict(X) #Compute cluster centers and predict cluster index for each sample.
#Convenience method; equivalent to calling fit(X) followed by predict(X).


print(X[:,0])
print(X[:,1])
# Plot the data
plt.scatter(X[:,0],
            X[:,1])   # Plotting the 2 dimensions points on x and y axis
print(kmeans.cluster_centers_[:, 0])
print(kmeans.cluster_centers_[:, 1])

print(kmeans.cluster_centers_)
# Plot the clusters on top of the data
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=100,                             # Set centroid size
            c='green')                           # Set centroid color
plt.show()
