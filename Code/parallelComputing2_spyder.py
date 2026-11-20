from pathlib import Path

#Final Code - Clustering V5

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from astropy.io import ascii
from astropy.table import Table, vstack
from astropy.table import join
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

#Limit amount of threads due to limitations with kmeans (3 default)
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

perp = 30 #30 default
error = 4 #medelvärde (DEFAULT = 4)

DATA_DIR = Path(__file__).resolve().parent

# Read in file (original file)
table = ascii.read(str(DATA_DIR / "tablec3.dat"),
                   readme=str(DATA_DIR / "ReadMe"))

#Read Batistini file

table2 = ascii.read(str(DATA_DIR / "table5.dat"),
                   readme=str(DATA_DIR / "ReadMe2"))

# Show available columns

# Merge tables based on HIP number
table = join(table, table2, keys="HIP", join_type="left")  # 'left' keeps all HIP from table

# Show available columns
print(table.colnames)

# Check the first few rows to verify the alignment of HIP values
#print("\nFirst 6 rows of merged table:")
#print(table[:6])

#print("B_Age TEST: ",table["B_Age"][0])
#print("b_Age TEST: ",table["b_Age"][0])
#print("Age TEST: ",table["Age"][0])

def gauss(x,amp,mu,sigma): #gaussian
    return amp * np.exp(-(x-mu)**2/2*sigma**2)

def MGFE(perfectIndicies,bad_indicies2):
    #Mg/Fe vs Age

    mg_fe = []
    mg_fe_err = []
    age = []
    age_err = []

    for i in range(len(perfectIndicies)):
        mg_fe.append(table["Mg/Fe"][perfectIndicies[i]])
        age.append(table["Age"][perfectIndicies[i]])

    for i in range(len(bad_indicies2)):
        mg_fe_err.append(table["Mg/Fe"][bad_indicies2[i]])
        age_err.append(table["Age"][bad_indicies2[i]])

    plt.scatter(age,mg_fe,c=age, cmap="seismic", vmin=np.min(age), vmax=np.max(age))
    plt.scatter(age_err,mg_fe_err,color = "grey")

    plt.xlabel("Age [Gyr]")
    plt.ylabel("Mg/Fe")
    plt.show()

def Histograms(checkVals,_tsneX,reverse_index_map, labels, inliers):

    #HISTOGRAMS

    for w in range(len(checkVals)):
        histvals = []
        for q in range(len(_tsneX)):
            histvals.append(table[checkVals[w]][reverse_index_map[q]])

        histvals = np.array(histvals)

        print("Length Histogram (Total): ", len(histvals))
        
        cluster1_hist = histvals[labels == 0]
        cluster2_hist = histvals[labels == 1]

        print("Length Histogram (CLUSTER 1): ", len(cluster1_hist))
        print("Length Histogram (CLUSTER 2): ", len(cluster2_hist))

        #Error filtering included from previous t-SNE method
        title1 = "Histograms for overdensities " + str(checkVals)
        title2 = "Cumulative Histograms for overdensities " + str(checkVals)
        plt.hist(cluster1_hist,bins = "auto",color="blue", alpha=0.5)
        plt.hist(cluster2_hist,bins = "auto",color= "red", alpha=0.5)
        plt.ylabel(checkVals[w])
        plt.xlabel("N")
        plt.title(title1)
        plt.show()

        plt.hist(cluster1_hist,bins = "auto",color="blue",cumulative= True, density=True,alpha=0.5)
        plt.hist(cluster2_hist,bins = "auto",color= "red",cumulative= True, density=True,alpha=0.5)
        plt.ylabel(checkVals[w])
        plt.xlabel("N")
        plt.title(title2)
        plt.show()


    #Filtered HISTOGRAMS
    for w in range(len(checkVals)):
        histvals = []
        for q in range(len(_tsneX)):
            histvals.append(table[checkVals[w]][reverse_index_map[q]])

        histvals = np.array(histvals)

        print("Length Histogram (Total): ", len(histvals))

        # Apply inliers filter
        histvals_filtered = histvals[inliers]
        labels_filtered = labels[inliers]

        # Separate by cluster after filtering
        cluster1_hist = histvals_filtered[labels_filtered == 0]
        cluster2_hist = histvals_filtered[labels_filtered == 1]

        print("Length Histogram (CLUSTER 1): ", len(cluster1_hist))
        print("Length Histogram (CLUSTER 2): ", len(cluster1_hist))

        # Plot standard histogram
        title1 = f"Filtered Histograms for overdensity {checkVals[w]}" + " TEST"
        plt.hist(cluster1_hist, bins="auto", color="blue", alpha=0.5, label="Cluster 1")
        plt.hist(cluster2_hist, bins="auto", color="red", alpha=0.5, label="Cluster 2")
        plt.ylabel(checkVals[w])
        plt.xlabel("N")
        plt.title(title1)
        plt.legend()
        plt.show()

        # Plot cumulative histogram
        title2 = f"Cumulative Filtered Histograms for overdensity {checkVals[w]}" + " TEST"
        plt.hist(cluster1_hist, bins="auto", color="blue", cumulative=True, density=True, alpha=0.5, label="Cluster 1")
        plt.hist(cluster2_hist, bins="auto", color="red", cumulative=True, density=True, alpha=0.5, label="Cluster 2")
        plt.ylabel(checkVals[w])
        plt.xlabel("N")
        plt.title(title2)
        plt.legend()
        plt.show()

def ClusterPlot(filteredAges,labels,kmeans,inliers,threshold, X):
    cluster1 = X[labels == 0]
    cluster2 = X[labels == 1]

    #Ages color
    ages1 = filteredAges[labels == 0]
    ages2 = filteredAges[labels == 1]

 
    # Normalize color map globally across all ages
    norm = Normalize(vmin=filteredAges.min(), vmax=filteredAges.max())
    cmap = plt.cm.seismic


    plt.scatter(cluster1[:,0], cluster1[:,1], label = "Cluster 1", c= ages1,norm=norm, cmap = cmap, marker='o', alpha=0.8, edgecolors='k', linewidths=0.3)
    plt.scatter(cluster2[:,0], cluster2[:,1], label = "Cluster 2",c=ages2,norm=norm, cmap = cmap, marker='^', alpha=0.8, edgecolors='k', linewidths=0.3)   
    plt.title("Cluster Plot ")
    plt.show()

  


    # Only keep inliers for plotting
    X_filtered = X[inliers]
    filtered_ages = filteredAges[inliers]
    labels_filtered = labels[inliers]

    cluster1test = X_filtered[labels_filtered == 0]
    cluster2test = X_filtered[labels_filtered == 1]

    ages1test = filtered_ages[labels_filtered == 0]
    ages2test = filtered_ages[labels_filtered == 1]

 
    # Normalize color map globally across all ages
    norm = Normalize(vmin=filteredAges.min(), vmax=filteredAges.max())
    cmap = plt.cm.seismic

    txtTitle = "Cluster Plot " + str(threshold) + " threshold"
    plt.scatter(cluster1test[:,0], cluster1test[:,1], label = "Cluster 1", c= ages1test,norm=norm, cmap = cmap, marker='o', alpha=0.8, edgecolors='k', linewidths=0.3)
    plt.scatter(cluster2test[:,0], cluster2test[:,1], label = "Cluster 2",c=ages2test,norm=norm, cmap = cmap, marker='^', alpha=0.8, edgecolors='k', linewidths=0.3)   
    plt.title(txtTitle)
    plt.show()

def Cluster(_tsneX,_tsneY, index_map, checkVals, mappedPerfect, filteredAges):
    global error
    reverse_index_map = {v: k for k, v in index_map.items()}

    #print("Index map: ", index_map)
    #print("Reverse index map: ", reverse_index_map)

    X = np.column_stack((_tsneX, _tsneY))  # Shape: (n_samples, 2)

    #kmeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init = "auto").fit(X)
    labels = kmeans.labels_

    inertia = kmeans.inertia_ #Lower inertia = tighter clusters = better performance

    print("Inertia (Clustering Performance): ", str(inertia))

    #Filtering "NOISE STARS"

    # Step 2: Distance to assigned cluster center
    #_, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)

    # Step 3: Threshold (e.g. 70th percentile)
    #threshold = np.percentile(distances, 30)
    #inliers = distances < threshold  # Boolean mask

    #Visualization (OFF for optimization)
    #ClusterPlot(filteredAges,labels,kmeans,inliers,threshold,X)

    #Histograms(checkVals,_tsneX,reverse_index_map,labels,inliers)

    return inertia

def OriginalPlot(dim, ages,txtTitle,tsneX,tsneY,filtered_ages,tsneXerr,tsneYerr, X_embedded_3D, X_embedded):
    if dim == 0 or dim == 2:
        # 2D Scatter Plot with seismic colormap
        #plot everything
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=ages, cmap="seismic", vmin=np.min(ages), vmax=np.max(ages))

    
        plt.colorbar(scatter, label="Age")  # Use full range of ages
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        plt.title(txtTitle)
        plt.show()
        
        # Plot good points (colored by age)
        scatter_good = plt.scatter(tsneX, tsneY, c=filtered_ages, cmap="seismic", vmin=np.min(filtered_ages), vmax=np.max(filtered_ages))

        # Plot bad points (grey, without affecting the colorbar)
        plt.scatter(tsneXerr, tsneYerr, color="grey")

        # Add colorbar using the correct scatter
        plt.colorbar(scatter_good, label="Age")

        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        plt.title(txtTitle)
        plt.show()

    if dim == 1 or dim == 2:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')  # Define 3D projection
        sc = ax.scatter(X_embedded_3D[:, 0], X_embedded_3D[:, 1], X_embedded_3D[:, 2], c=ages, cmap="seismic", vmin=np.min(ages), vmax=np.max(ages))

        ax.set_xlabel("TSNE Component 1")
        ax.set_ylabel("TSNE Component 2")
        ax.set_zlabel("TSNE Component 3")
        ax.set_title(txtTitle)

        # Add color bar
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label("Age")

        plt.show()

        #with error color grading

        #fig = plt.figure(figsize=(8, 6))
        #ax = fig.add_subplot(111, projection='3d')  # Define 3D projection
        #sc = ax.scatter(tsneX, tsneY, tsneZ, c=filtered_ages, cmap="seismic", vmin=np.min(filtered_ages), vmax=np.max(filtered_ages))
        #ax.scatter(tsneXerr, tsneYerr, tsneZerr, color = "grey")
        #ax.set_xlabel("TSNE Component 1")
        #ax.set_ylabel("TSNE Component 2")
        #ax.set_zlabel("TSNE Component 3")
        #ax.set_title(txtTitle)

        # Add color bar
        #cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        #cbar.set_label("Age")

        #plt.show()
      
def tsne_plot(checkValues, dim, include_velocity=False):
    global error
    source_size = len(table)  # Number of stars in the dataset
    alpha = [table[col].data for col in checkValues]  # List of lists

    #print("Abundance Values:")
    #for col, values in zip(checkValues, alpha):
        #print(f"{col}: {values[:5]} ...")  # Print first 5 values for brevity
   
    omega = np.array(alpha).T  # Shape: (source_size, features)

    # Identify rows with missing values using .mask
    bad_indices = set()
    for i in range(source_size):  # Loop over rows
        for col in checkValues:
            if hasattr(table[col], 'mask') and table[col].mask[i]:  # Check for masked values
                bad_indices.add(i)  # Store row index
                break  # No need to check further in this row
    
    # Extracting ages
    ages = np.array(table["Age"].data)  # Convert to NumPy array

    # Compute `sqrt(Ulsr^2 + Wlsr^2)` if `include_velocity=True`
    if include_velocity:
        ulsr = np.array(table["Ulsr"].data)
        wlsr = np.array(table["Wlsr"].data)
        velocity_magnitude = np.sqrt(ulsr**2 + wlsr**2)

        # Check for missing values in Ulsr or Wlsr
        for i in range(source_size):
            if (hasattr(table["Ulsr"], 'mask') and table["Ulsr"].mask[i]) or \
               (hasattr(table["Wlsr"], 'mask') and table["Wlsr"].mask[i]):
                bad_indices.add(i)  # Exclude these indices

    # Get indices of valid (non-missing) rows
    good_indices = np.array([i for i in range(source_size) if i not in bad_indices])

    #print("good inidicies: ", good_indices)
    #Filtering based on error in ages
    print("Filtered values after missing elements: ", len(good_indices))

        
    # Filter `omega`, `ages`, and velocity if included
    omega = omega[good_indices, :]
    ages = ages[good_indices]

    if include_velocity:
        velocity_magnitude = velocity_magnitude[good_indices]
        omega = np.column_stack((omega, velocity_magnitude))  # Add velocity feature

    # Print for verification
    print(f"Original stars: {source_size}, After filtering: {len(ages)}")

    

    global perp  # perplexity

    # Apply TSNE
    if dim == 0 or dim == 2:
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perp, random_state=42).fit_transform(omega)
    #if dim == 1 or dim == 2: 
        # 3D TSNE Projection with seismic colormap
        #X_embedded_3D = TSNE(n_components=3, learning_rate='auto', init='pca', perplexity=perp, random_state=42).fit_transform(omega) 

    if include_velocity:
        txtTitle = "TSNE Projection of " + str(checkValues) + " + sqrt(u^2 + w^2)"
    else:
        txtTitle = "TSNE Projection of " + str(checkValues)


    #Error filtering
    perfect_indicies = [] #indicies within error age limit [BEFORE TSNE]
    bad_indices2 = [] #indicies after age errors [BEFORE TSNE]

    for i in range(len(good_indices)):
        #t_age = table["Age"][good_indices[i]] 
        t_top = table["B_Age"][good_indices[i]] 
        t_down = table["b_Age"][good_indices[i]] 

        t_test = (t_top-t_down)/2
        if t_test < error:
            perfect_indicies.append(good_indices[i])
        else:
            bad_indices2.append(good_indices[i])
        

    print("Filtered values after error limit: ", len(perfect_indicies))
    print("Size Perfect Indicies: ", len(perfect_indicies))
    print("Size bad Inidicies: ", len(bad_indices2))

    #print("Good indicies: ", good_indices)
    #print("Perfect indicies", perfect_indicies)

    # Create a mapping from original index to new index in X_embedded
    index_map = {good_indices[i]: i for i in range(len(good_indices))} 

    #reverse_index_map = {v: k for k, v in index_map.items()}
    #print("Index map", index_map)
    #print("Reverse Index map ", reverse_index_map)

    # Convert original indices to t-SNE indices using the mapping
    mapped_perfect = [index_map[idx] for idx in perfect_indicies if idx in index_map]
    mapped_bad = [index_map[idx] for idx in bad_indices2 if idx in index_map]

    # Filter `ages` using mapped indices
    filtered_ages = ages[mapped_perfect]  # Ensure ages matches tsneX and tsneY

    #print("Filtered ages: ", filtered_ages)

    tsneX = []
    tsneY = []
    #tsneZ = []

    tsneXerr = []
    tsneYerr = []
    #tsneZerr = []


    # Extract X and Y coordinates using NumPy indexing
    tsneX = X_embedded[mapped_perfect, 0].tolist()
    tsneY = X_embedded[mapped_perfect, 1].tolist()
    tsneXerr = X_embedded[mapped_bad, 0].tolist()
    tsneYerr = X_embedded[mapped_bad, 1].tolist()

    # Handle the Z dimension conditionally
    #if dim == 1 or dim == 2:
        #tsneZ = X_embedded[mapped_perfect, 2].tolist()
        #tsneZerr = X_embedded[mapped_bad, 2].tolist()


    

    #Plotting

    #OriginalPlot(dim,ages,txtTitle,tsneX,tsneY,filtered_ages,tsneXerr,tsneYerr,X_embedded_3D, X_embedded)

    #Cluster + Histograms

    inertia = Cluster(tsneX,tsneY,index_map,checkValues,mapped_perfect,filtered_ages)
    

    #Mg/Fe plots

    #MGFE(perfect_indicies,bad_indices2)

    return inertia


# Your list of features
allCheckValues = ['Fe/H', 'O/Fe', 'Na/Fe', 'Mg/Fe', 'Al/Fe', 'Si/Fe', 'Ca/Fe', 'Ti/Fe', 'Cr/Fe', 'Ni/Fe', 'Zn/Fe', 
                  'Y/Fe', 'Ba/Fe', 'Ti/H', 'O/Ti', 'Na/Ti', 'Mg/Ti', 'Al/Ti', 'Si/Ti', 'Ca/Ti', 'Cr/Ti', 
                  'Ni/Ti', 'Fe/Ti', 'Zn/Ti', 'Y/Ti', 'Ba/Ti', '[Sc/H]', '[Mn/H]']

# Function for parallel execution
def evaluate_combo(combo):
    input_array = list(combo)
    try:
        inertiaTest = tsne_plot(input_array, 2)
    except Exception as e:
        inertiaTest = np.inf  # Handle errors gracefully
    return inertiaTest, input_array

def main():
    print("Amount of input parameters:", len(allCheckValues))

    # Generate all 5-element combinations
    combinations = list(itertools.combinations(allCheckValues, 5))

    # Define number of cores (customize as needed)
    num_cores = os.cpu_count() - 1  # Use all available cores minus one (8 low performance)
    print(f"Using {num_cores} CPU cores...")

    inertia = np.inf
    resCombo = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(evaluate_combo, combo) for combo in combinations]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running t-SNE in parallel"):
            inertiaTest, input_array = future.result()
            if inertiaTest < inertia:
                inertia = inertiaTest
                resCombo = input_array

    print("Best combination:", resCombo)

# Required for multiprocessing on some platforms
if __name__ == "__main__":
    main()




