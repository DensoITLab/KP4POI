import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from node2vec_recursive_clustering import vectorization
from node2vec_recursive_clustering.node2vec_recursive_clustering import Node2Vec_Recursive_Clustering
import community


def c_part(g):
    partition = community.best_partition(g)

    with open('community.json','wt') as fout:
        json.dump(partition,fout,indent=2)


def get_emb(g):
    vec = vectorization.Vectorization()

    # set network
    vec.setg(g)

    # set parameter
    vec.set_parameter(dimensions=128,walk_length=5,num_walks=10,p=1,q=1,workers=4,window=15,min_count=1,sg=1)

    # conduct node2vec algorithm and transform to d-diimensional vectors
    vec.conduct_vectorization()

    # obtain d-dimensional vectors
    vec_df = vec.get_vec().sort_index()
    print(vec_df.head())

    return vec_df


def recursive_clustering(g,vec_df):

    dat = Node2Vec_Recursive_Clustering()

    # set data
    dat.set_existing(g,vec_df)

    # perform the first step
    dat.first_step(alpha=1.0,centrality_centroid=False,do_plot=True,spd_consideration=False,start=20,ns=10)

    # obtain the first step result
    first_res = dat.get_first_step()

    # save the result (you can set this reuslt with 'dat.set_first_step()' and skip the above)
    pd.to_pickle(first_res,'first_res.pkl')

    # move on to the recursive clstering step
    dat.recursive_step(min_threshold=20,rich_threshold=0.8,min_q=0.4,do_plot=False,centrality_centroid=False,ns=5)

    # summarize the result
    dat.summarize_module()

    # obtain dataframe contains module information
    module_df = dat.get_module()
    print(module_df.head(5))

    # save result
    module_df.to_csv('module_df.csv')

def kmeans(vec_df):
    print(vec_df.head)
    idx = vec_df.index
    vec = vec_df.to_numpy()
    print(vec.shape)

    #silhouette_analysis(vec,[10,20,50,100,200,500])

    kmeans = KMeans(n_clusters=50, random_state=0, n_init="auto").fit(vec)
    #np.save('labels.npy',kmeans.labels_)
    #labels_ = np.load('labels.npy')
    labels_ = kmeans.labels_

    labels = {}
    for i,j in enumerate(idx):
        labels[str(j)] = int(labels_[i])
    
    return labels

 

def silhouette_analysis(X,range_n_clusters):

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples, silhouette_score

    for n_clusters in range_n_clusters:
        print(f'n clusters {n_clusters}')
        
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d (score = %.3f)"
            % (n_clusters, silhouette_avg),
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(f'silhouette_analysis_c{n_clusters}.png')


def graph2vec(g):
    from karateclub.graph_embedding import Graph2Vec
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt

    model = Graph2Vec(wl_iterations=2, dimensions=256, 
                    down_sampling=0.0001, epochs=100, learning_rate=0.025, min_count=10)
    model.fit(g)
    with open('model.pkl','wb') as fout:
        pickle.dump(model,fout)


    with open('model.pkl','rb') as f:
        model = pickle.load(f)

    emb = model.get_embedding() # (1108, 128)
    print(emb)


    sns.clustermap(emb)
    plt.savefig('heatmap.png')

def plot_pca(feature):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig('pca.png')

def plot_cont_ratio(pca):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.figure(figsize=(6, 6))
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.tight_layout()
    plt.savefig('pca_cont.png')
