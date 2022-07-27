import os

import gav_oidium_const as gof

txt_oiv_452_spec_req = """
**From the specifications we now that a clean dataframe has the following rules**:
- _sporulation_ **must be** 1 ou 0
- if _sporulation_ **is** 0 , _densite_sporulation_ **must be** NaN else it **must be** an odd number
- _densite_sporulation_ **must be** a number and **not** 0
- _necrosis_ **must be** 1 ou 0
- if _necrosis_ **is** 1 _surface_necrosee_ & _taille_necrose_ **must not be** none else they **must**
- _surface_necrosee_ & _taille_necrose_ **must be** NaN or odd
- _OIV_ **must be** an odd number
- if _OIV_ is 9 **there must be no** _sporulation_ else **there must be**
- _ligne_ **must not** be NA
"""

txt_oiv_452_spec = """
### OIV
OIV 452-2 is a standard to evaluate resistance to powdery mildew in vine disk leafs

&ndash; From OIV the 452-2 specification.
>
>  Characteristic: Leaf: degree of resistance to Plasmopara (leaf disc test)  
>  Notes:
>  1: very little 3:little 5:medium 7:high 9:very high   
>  Observation during the whole vegetation period, as long as there are young leaves, on vines not treated with
>  chemicals.
>  Because the zoospores penetrate through the stomata, the leaf discs have to be placed with the lower surface up.
>  Using a standardized spore suspension with 25000 spores/ml (counting chamber), a pipette is used to place 40µl
>  or 1000 spores on each leaf disc.
>  Incubation: in complete darkness (aluminum coat), room temperature, 4 days.
>  Remark: if the inoculum remains on the leaf disc too long, lesions are produced. Therefore, 24 hours after
>  inoculation, the spore suspension has to be removed by blotting with a filter paper. 
"""

txt_target = """
**This notebook will**:
- Retrieve all available Excel files
- Translate them to CSV and merge them
- Build models to asses the possibility of predicting OIV from various visual variables

**We need**:
- Base python libraries for file management
- Pandas and Numpy for the dataframes
- SkLearn for statistics
- Plotly for ... plotting    
"""

txt_python_need = f"""
**Functions needed to**:
- Check that the dataframe has at least the needed columns
- Plot model variance
- Plot an histogram of the variables needed for the OIV so inconsistencies can be detected
- Generate categorical OIV from dataframe
**Constants**:
- Path to datain: {os.path.abspath(gof.datain_path)}
- Path to distant Excel files: {os.path.abspath(gof.excel_file_path)}
- Path to local EXcel files: {os.path.abspath(gof.oidium_extracted_csvs_path)}
- Path to extracted CSVs: {os.path.abspath(gof.excel_file_list_path)}
- Path to individual CSV generation result: {os.path.abspath(gof.path_to_df_result)}
- Needed columns: {gof.needed_columns}
"""

txt_get_excels = """
Get all related file's path in the distant server.  
Experiements are stored by year and by experiment, the excels files with data are Excel classifiers which contain "saisie", 
we're going to parse all the folders year by year and retrieve the files.

- Files containing DM for domny mildew, ie mildiou, are selected for OIV analysis
- Files containing PM for powdery mildew, ie oïdium, are discarded
"""

txt_excel_headers = """
We look for 2 particular headers, sheets will be discarded if:
- the header is not found
- the dataframe is corrupted, ie unable to find images or a column is malformed
"""

txt_rejected_csvs = """
**Some CSVs where rejected:**
- Some are experiment description with no data
- Some have no images
- Some are corrupted, ie, it was impossible to read them
- ...
"""

txt_fail = """
This has not been successful, were going o try switching from a resistance scale to a susceptibility scale, 
this allows us to keep all dimensions for all observations.  
If we invert the axes to have sensitivity scale instead of a resistance scale, 
this will allow us to include all previously removed NaN contained rows as all OIV 5 rows
"""

txt_kmeans = """
### Maybe OIV is not the best way to group observations

There's one of multiple cases happening here, but we're only going to analyze 1 and discuss the other. So either:
- The **variables** can't provide any information on the response of the plant to the pathogen, unlikely
- The **OIV** alone cannot give a good indication of the plant's resistance
As said we're only going to expand on the second option

That is why we're going to use k-means as a mean to see if the data can cluster without the **OIV**
We're going to cluster the data with K-means with a class count going from 2 to 10
"""

txt_kmeans_pca = """
#### PCA
Since the data has more than 3 dimensions it will be impossible to plot the result 
of the k-means clustering in a 2 or 3D plot. That is why we start by fitting a PCA to 
the data, then find how many components are needed to explain most of the variability. 
Finally we will use k-means on the **latent space*** of the PCA

&ndash; *Latent space, from [Wikipedia](https://en.wikipedia.org/wiki/Latent_space)
> A latent space, also known as a latent feature space or embedding space, is an embedding
> of a set of items within a manifold in which items which resemble each other more 
> closely are positioned closer to one another in the latent space. Position within 
> the latent space can be viewed as being defined by a set of latent variables that 
> emerge from the resemblances from the objects.
> In most cases, the dimensionality of the latent space is chosen to be lower than the 
> dimensionality of the feature space from which the data points are drawn, making 
> the construction of a latent space an example of dimensionality reduction, which 
> can also be viewed as a form of data compression or machine learning.
> A number of algorithms exist to create latent space embeddings given a set of data 
> items and a similarity function.

"""

txt_kmeans_elbow = """
#### Using the elbow method to find the optimal class count

&ndash; From [Wikipedia](https://en.wikipedia.org/wiki/Elbow_method_(clustering))

> In cluster analysis, the elbow method is a heuristic used in determining the number 
> of clusters in a data set. The method consists of plotting the explained variation 
> as a function of the number of clusters and picking the elbow of the curve as the 
> number of clusters to use. The same method can be used to choose the number of 
> parameters in other data-driven models, such as the number of principal components 
> to describe a data set.

> The method can be traced to speculation by Robert L. Thorndike in 1953.[1]


"""

txt_kmeans_silhouette = """
#### Using the silhouette method to find the optimal class count

&ndash; From [Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering))

> Silhouette refers to a method of interpretation and validation of consistency within 
> clusters of data. The technique provides a succinct graphical representation of how 
> well each object has been classified.[1] It was proposed by Belgian statistician 
> Peter Rousseeuw in 1987.
> The silhouette value is a measure of how similar an object is to its own cluster 
> (cohesion) compared to other clusters (separation). The silhouette ranges 
> from −1 to +1, where a high value indicates that the object is well matched to its 
> own cluster and poorly matched to neighboring clusters. If most objects have a high 
> value, then the clustering configuration is appropriate. If many points have a low or 
> negative value, then the clustering configuration may have too many or too few clusters.

> The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance.

"""

txt_kmeans_explore_cluster_count = """
#### k-means for cluster count 2 to 9
As we don't know how many cluster are in the data we're going first to explore a range visually 
"""

txt_kmeans_what = """
#### What does all this mean and what do we do now
Three results come from the k-means, the elbow and the silhouette:
- Visually it looks like there are 3 clusters at most that can be distinguished.
- The elbow method shows 6
- The silhouette shows a maximum coefficient value for 8 clusters, but there's not a big difference between the counts

##### Intercluster Distance Maps
To try and find an explanation we're going to check th **Intercluster Distance Maps*** to 
see if some observations are too far away from the cluster's centers

&ndash; *Intercluster Distance Maps, from [YellowBrick](https://www.scikit-yb.org/en/latest/api/cluster/icdm.html)

> Intercluster distance maps display an embedding of the cluster centers in 2 
> dimensions with the distance to other centers preserved. E.g. the closer to centers 
> are in the visualization, the closer they are in the original feature space. 
> The clusters are sized according to a scoring metric. By default, they are sized 
> by membership, e.g. the number of instances that belong to each center. This 
> gives a sense of the relative importance of clusters. Note however, that because 
> two clusters overlap in the 2D space, it does not imply that they overlap in the 
> original feature space.
"""

txt_kmeans_3D = """
##### 3D projections of k-means models
The results are confusing, 6 and 8 are supposed to be the best but they don't look that good.
Maybe we should see the data in a 3D projection?
"""

txt_noiv_models = """
### Models with NOIVs
From the k-means we now know that it may be better to discriminate our observations with 3
new classes.

#### sPLS-DA's with the three new NOIVs
"""

txt_noiv_svm = """
#### SVM Classification
"""
