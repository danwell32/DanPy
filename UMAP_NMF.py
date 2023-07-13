### Funciones de este modulo: 
# def compute_umap(data, n_comp=2, dlist=[1.,0.75,0.5,0.25], mlist=[25,50,100,150])
# def visualize_umap(umap_dict_mask, min_dist_values, n_neighbors_values)
# def matrix_factor_DimRed(ncompo, algo, data_matrix, Eloss, archetypes = True, graphs = None, poisson = False, ncols = 6)
# def kmeans_clustering(matrix, n_cluster, norma='l2')
# def get_cmap(labels, noise = 'grey',paleta = 'Spectral')
# def export_svg(obj, filename)

# Modulos importados: 
import numpy as np
from time import time
import copy as cp
import hyperspy.api as hs


#Xarray:
import xarray as xr

#Matplotlib: 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Sklearn: 
import sklearn
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

#Holoviews, Bokeh and panel:
import holoviews as hv
import panel as pn
import bokeh
from bokeh import palettes
from bokeh.io import export_svgs
from bokeh.io import show

#UMAP:
import umap

#HDBSCAN:
import hdbscan

#-------------------------------------------------------------------------------------------------------------------#
def kmeans_clustering(matrix, n_cluster, norma='l2'):
    '''
    Función que aplica kmeans clustering en la imagen de espectros.
    
    Parameters:
    -----------
    matrix: numpy array. (x,y,eloss)
        Imagen de espectros.
    n_cluster: int.
        Número de clusters.
    norma: string, optional. (default='l2')
        Normalización que queremos aplicar. Opciones: ‘l1’, ‘l2’, ‘max’ or None.
        
    Returns:
    --------
    labels: numpy array. (x,y)
        Matriz con las etiquetas de cada cluster. 
    centres: numpy array. (n_cluster,eloss)
        Matriz que contiene los centroides de cada cluster identificado (estos centroides provienen de los espectros normalizados). 
    '''
    allowed_norms = ['l1', 'l2', 'max', None]
    if norma not in allowed_norms:
        raise ValueError(f"norma debe ser uno de {allowed_norms}")
        
    matrix_norm = matrix.copy()
    matrix_norm = matrix_norm.reshape(matrix.shape[0]*matrix.shape[1], matrix.shape[-1])
    if norma is None: 
        sclust_norm = matrix_norm
    else:
        sclust_norm, _ = normalize(matrix_norm,norm=norma,axis=1,return_norm=True)
    kmeans = KMeans(n_clusters=n_cluster, tol=1e-9, max_iter=700)
    fitted = kmeans.fit(sclust_norm)
    centres = fitted.cluster_centers_
    labels = fitted.labels_.reshape(matrix.shape[:-1])
    return labels, centres
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def compute_umap(data, n_comp=2, dlist=[1.,0.75,0.5,0.25], mlist=[25,50,100,150]):
    """
    Compute UMAP dimensional reduction on the input data. 

    Parameters:
    data (np.array): Array of spectral data to be reduced in dimensionality.
    n_comp (int, optional): Number of components for the UMAP dimensional reduction. Defaults to 2.
    dlist (list, optional): List of `min_dist` parameter values to be tried in the UMAP algorithm. Defaults to [1.,0.75,0.5,0.25].
    mlist (list, optional): List of `min_neigh` parameter values to be tried in the UMAP algorithm. Defaults to [25,50,100,150].

    Returns:
    umap_data_all (list): List of arrays, each containing the UMAP reduced data for a combination of `min_dist` and `min_neigh` parameters.
    udat_dict (dict): Dictionary of UMAP objects, each object being associated with a combination of `min_dist` and `min_neigh` parameters. 
    The keys are formatted as 'udata_{}_{}', where the first value is `min_dist` and the second value is `min_neigh`.
    """

    data_matrix_all = np.array(data)

    # Verificar si la dimension de los datos es mayor a 2
    if len(data_matrix_all.shape)<=2: 
        pass
    elif len(data_matrix_all.shape)==3:
        x = data_matrix_all.shape[0]
        y = data_matrix_all.shape[1]
        eloss = data_matrix_all.shape[2]
        # Re-estructurar los datos en una matriz de dos dimensiones
        data_matrix_all = data_matrix_all.reshape(x*y,eloss)
    else:
        raise ValueError(f"The dimensions of the data array provided {len(data_matrix_all.shape)} are larger than 3.")
    
    # Definimos los objetos para guardar los datos: 
    umap_data_all = []
    udat_dict = dict()
    # Tiempo de ejecucion:
    times_l = []

    # Iterar sobre los valores de dlist y mlist
    for d in dlist:
        for m in mlist:
            print('min_dist = {} and min_neigh = {}'.format(d,m))
            t0 = time()
            # Crear el objeto UMAP
            mapper = umap.UMAP(n_neighbors = m,
                            random_state = 1,
                            n_components = n_comp,
                            min_dist = d)
            # Realizar la reduccion de dimensional: 
            umap_data_all.append(mapper.fit_transform(data_matrix_all))
            udat_dict['udata_{}_{}'.format(d,m)] = mapper
            t1 = time()
            times_l.append(round(t1-t0,2))
            print('Tiempo transcurrido: {} s'.format(round(t1-t0,2)))
    return umap_data_all, udat_dict
#-------------------------------------------------------------------------------------------------------------------#

#Visualizar los datos de umap con hollowviews: 
#-------------------------------------------------------------------------------------------------------------------#
def visualize_umap(umap_dict_mask, min_dist_values, n_neighbors_values):
    """
    Genera visualizaciones UMAP para diferentes combinaciones de min_dist y n_neighbors a partir de un diccionario de modelos UMAP ajustados.
    
    Parameters
    ----------
    umap_dict_mask : dict
        Diccionario de modelos UMAP ajustados, con claves en el formato 'udata_{min_dist}_{n_neighbors}'.
        
    min_dist_values : list of float
        Lista de valores min_dist para los que se generarán visualizaciones UMAP.
        
    n_neighbors_values : list of int
        Lista de valores n_neighbors para los que se generarán visualizaciones UMAP.
        
    Returns
    -------
    layout : holoviews.core.layout.Layout
        Un objeto holoviews Layout que contiene las visualizaciones UMAP para cada combinación de min_dist y n_neighbors.
    """
    embed_mask = []
    for i in min_dist_values:
        for j in n_neighbors_values:
            zers = np.zeros((umap_dict_mask[f'udata_{i}_{j}'].embedding_.shape[0], 3))
            zers[:, :-1] = umap_dict_mask[f'udata_{i}_{j}'].embedding_
            embed_mask.append(hv.Points(zers, vdims=['color'])\
                .opts(frame_width=650, frame_height=300, toolbar=None, fill_alpha=0.5, bgcolor='black',
                      line_alpha=0, line_width=0.15, size=2.5, xaxis=None, yaxis=None,
                      show_legend=True, color='color', shared_axes=False,
                      title=f'UMAP on masked data, min_dist={i}, n_neighbors={j}'))
    
    return hv.Layout(embed_mask).cols(len(n_neighbors_values))
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def matrix_factor_DimRed(ncompo, algo, data_matrix, Eloss, archetypes = True, graphs = None, poisson = False, ncols = 6):
    '''
    This function is in charge of carrying out the matrix factorization of our dataset
    - Parameters -
    ncompo : int. Number of components for the matrix factorization
    algo: str. Algorithm for the matrix factorization. It will only allow the PCA and NMF
               implementations in sklearn. In the PCA case, the whiten option for the 
               loadings results is availbale (see the sklearn documentation to know more)
               Values: 'NMF', 'PCA-w', 'PCA'.
    data_matrix: np.array. Matrix of the spectrum image data. The shape must be (x,y,Eloss)
                 The inner workings of the function will take care of the flattening for the 
                 sklearn algorithms.
    Eloss : np.array. Energy loss axis for the component representation
    archetypes: bool. Controls if the function returns the archetypes resolved (the factors)
    graphs : int. Controls whether or not the visual representation is launched upon running, and what
                  style is selected.
                  None - no graph representation
                  1 - Panel of loadings only
                  2 - list of loadings with the reference signals
                  3 - panel of loadings and overlay of reference signals
    poisson: bool. Controls if a Anscombe transform is applied to the dataset previous to the matrix fact.
    ncols: int. In case of a visual representation, controls the number of columns displayed
    
    '''
    data_matrix_i = cp.deepcopy(data_matrix)
    
    if algo == 'NMF':
        if np.min(data_matrix_i) < 0:
            data_matrix_i -= np.min(data_matrix_i)
        model = NMF(n_components=ncompo,max_iter=1500,tol = 1E-4)
    elif algo == 'PCA-w':
        model = PCA(n_components=ncompo,svd_solver='randomized',whiten=True)
    elif algo == 'PCA':
        model = PCA(n_components=ncompo,svd_solver='randomized',whiten=False)
    else:
        print('invalid option')
        return
    if poisson: data_matrix_i = np.sqrt(data_matrix_i+3/8)
    else: pass
    #We time it now
    t0 = time()
    D = model.fit_transform(data_matrix_i.reshape(-1,data_matrix_i.shape[-1]))
    t1 = time()
    t_ellapsed = round(t1-t0,2)
    H = model.components_
    #plotting
    D_images = D.reshape(data_matrix_i.shape[:-1]+(ncompo,))
    if graphs == 1:
        lista_ims = []
        for i in range(ncompo):
            lista_ims.append(\
                hv.Image(\
                    xr.Dataset({'Component {}'.format(i):(['x','y'],D_images[:,:,i])},\
                    coords = {'x':np.arange(D_images.shape[0]),'y':np.arange(D_images.shape[1])}),\
                kdims = ['y','x'])\
                .opts(title = 'Component {}'.format(i),\
                    shared_axes = False,invert_yaxis = True,aspect= 'equal',\
                    xaxis = None,yaxis = None,frame_height = 175,\
                    toolbar = None,colorbar = True,colorbar_position = 'bottom'))
        pn.Column(pn.pane.Markdown('## {} - {} components. **Ellapsed time {}s**'.\
                format(algo,ncompo,t_ellapsed)),\
                  pn.GridBox(*lista_ims,ncols=ncols)).show(threaded = True)
    elif graphs == 2:
        lista_ims = []
        for i in range(ncompo):
            imi = hv.Image(\
                xr.Dataset({'Component {}'.format(i):(['x','y'],D_images[:,:,i])},\
                coords = {'x':np.arange(D_images.shape[0]),'y':np.arange(D_images.shape[1])}),\
            kdims = ['y','x'])\
            .opts(title = 'Component {}'.format(i),\
                shared_axes = False,invert_yaxis = True,aspect= 'equal',\
                xaxis = None,yaxis = None,frame_height = 200,\
                toolbar = None,colorbar = True,colorbar_position = 'bottom')

            curve = hv.Curve(\
                xr.Dataset({'Archetype_{}'.format(i):(['Eloss'],H[i,:])},\
                coords={'Eloss':Eloss})).\
            opts(title='Archetype_{}'.format(i),xlabel = 'Electron Energy Loss[eV]',\
                ylabel = 'Counts[a.u]',frame_height= 200,frame_width= 500,\
                show_grid = True,shared_axes= False, framewise = True,toolbar = None)       
            lista_ims.append(pn.Row(imi,curve))

        pn.Column(pn.pane.Markdown('## {} - {} components. **Ellapsed time {}s**'.\
                format(algo,ncompo,t_ellapsed)),\
                  pn.GridBox(*lista_ims,ncols=ncols)).show(threaded = True)
    elif graphs == 3:
        lista_ims = []
        dictio_curves = dict()
        for i in range(ncompo):
            imi = hv.Image(\
                xr.Dataset({'Component {}'.format(i):(['x','y'],D_images[:,:,i])},\
                coords = {'x':np.arange(D_images.shape[0]),'y':np.arange(D_images.shape[1])}),\
            kdims = ['y','x'])\
            .opts(title = 'Component {}'.format(i),\
                shared_axes = False,invert_yaxis = True,aspect= 'equal',\
                xaxis = None,yaxis = None,frame_height = 200,\
                toolbar = None,colorbar = True,colorbar_position = 'bottom')

            curve = hv.Curve(\
                xr.Dataset({'Archetype_{}'.format(i):(['Eloss'],H[i,:])},\
                coords={'Eloss':Eloss})).\
            opts(title='Archetype_{}'.format(i),xlabel = 'Electron Energy Loss[eV]',\
                ylabel = 'Counts[a.u]',frame_height= 700,frame_width= 1000,\
                show_grid = True,shared_axes= False, framewise = True,toolbar = None)  
            lista_ims.append(imi)
            dictio_curves['Archetype {}'.format(i)] = curve

        pn.Column(pn.pane.Markdown('## {} - {} components. **Ellapsed time {}s**'.\
                format(algo,ncompo,t_ellapsed)),\
                  pn.GridBox(*lista_ims,ncols=ncols),\
                 pn.pane.HoloViews(hv.NdOverlay(dictio_curves).opts(legend_position = 'right')))\
        .show(threaded = True)

    else: pass
    if archetypes:
        return D,H
    else:
        return D
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def export_svg(obj, filename):
    """
    Exporta un objeto de Holoviews a un archivo .svg.

    Parameters:
        obj (Holoviews object): Objeto de Holoviews a exportar.
        filename (str): Nombre del archivo de salida.
    """
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'svg'
    export_svgs(plot_state, filename=filename)
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def select_palettes(paleta, num_labels):
    max_colors_spectral = 11
    if paleta == 'Spectral' and num_labels > max_colors_spectral:
        return 'Viridis'
    return paleta
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def get_cmap(labels, noise='grey', paleta='Spectral'):
    '''
    This method gets the labels introduced resolved by clustering algorithm
    and devices a colormap suitable to correctly represent the label-map
    - Parameters-
    labels : np.array. Labels from a fitted clustering model
    paleta : str(). Name of the colormap to be used as base
    - Return -
    cmap_db : list. Colormap list to be used by Holoviews in the label-maps
    limis   : tuple. Limits for the colorbar in the Holoviews Image for the label-map
    '''
    cmap_db = []
    num_labels = np.unique(labels).size 
    mask = 0
    
    # Check if only -1 and -2 are present
    if set(np.unique(labels)) == {-1}:
        if noise == 'grey':
            cmap_db.append('lightgrey')
        else:
            cmap_db.append('magenta')
        limis = (-0.5, 0.5)
        print('Solo -1')
        return cmap_db, limis
    else:
        pass
    
    # Add 'black' color for mask and 'grey' or 'magenta' for noise
    if -2 in np.unique(labels):
        cmap_db.append('black')
        num_labels -= 1
        mask -= 1
        print('Hay fondo',num_labels)
    else:
        pass
    
    if -1 in np.unique(labels):
        num_labels -= 1 
        print('Hay ruido',num_labels)
        if noise == 'grey': cmap_db.append('lightgrey')
        else: cmap_db.append('magenta')
        paleta = select_palettes(paleta, num_labels)     
        
        if num_labels <= 12 and num_labels > 3:
            cmap_db.extend(list(palettes.all_palettes[paleta][num_labels]))
            
        elif num_labels == 2:
            cmap_db.append('red')
            cmap_db.append('navy')

        elif num_labels == 3:
            cmap_db.append('red')
            cmap_db.append('navy')
            cmap_db.append('limegreen')

        elif np.max(labels) > 12:
            pal = list(palettes.all_palettes['Turbo'][256])
            if len(np.unique(labels)) > 256:
                raise Exception('Too many labels.. check if the clustering is well done')
            else:
                nlabs = len(np.unique(labels))
                cmap_db = [el[0] for el in np.array_split(pal, nlabs)]
        else:
            cmap_db.append('indigo')
            
        limis = (-1.5 + mask, len(cmap_db) - 1.5 + mask)
    else:
        cmap_db = []
        num_labels = np.unique(labels).size
        print('Ni fondo, ni ruido',num_labels)
        paleta = select_palettes(paleta, num_labels)
        if np.unique(labels).size <= 11 and np.unique(labels).size > 2:
            cmap_db.extend(list(palettes.all_palettes[paleta][np.unique(labels).size]))
        elif np.unique(labels).size == 2:
            cmap_db.extend(['lightgrey', 'dimgrey'])
        elif np.max(labels) >= 12:
            #It is unlikely that this would go beyond 256
            pal = list(palettes.all_palettes['Turbo'][256])
            if len(np.unique(labels)) > 256:
                raise Exception('Too many labels.. check if the clustering is well done')
            else: 
                nlabs = len(np.unique(labels))
                cmap_db = [el[0] for el in np.array_split(pal,nlabs)]
        else:
            cmap_db.append('indigo')
        limis = (-0.5, len(cmap_db) - 0.5)

    num_labels = np.unique(labels).size 
    assert len(cmap_db) == num_labels, f"Number of colors in palette ({len(cmap_db)}) does not match number of labels ({num_labels})"

    return cmap_db, limis
#-------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
def hdbscan_for_umap(umap_dict, min_dis, n_neigh, spectrum_image, eje = None, mask = None, filename = None):
    '''
    This function performs the HDBSCAN clustering algorithm on UMAP transformed data.
    It allows to manually adjust UMAP and HDBSCAN parameters, evaluate HDBSCAN values,
    assign outliers to clusters, and finally, visualize the results.
    
    Parameters:
    umap_dict: dict 
        Dictionary with UMAP embeddings.
    min_dis: float 
        The desired value for minimum distance parameter of UMAP.
    n_neigh: int 
        The desired value for number of neighbors parameter of UMAP.
    spectrum_image: array
        The image of spectra.
    eje: array
        An axis for the spectrum image.
    mask: array
        A mask for the image of spectra.
    filename: str
        Filename for saving the results.
        
    Returns:
    tuple: 
        Tuple containing matplotlib figures and labels.
    '''
    
    dpi = 150
    
    if isinstance(spectrum_image, hs.signals.EELSSpectrum):
        if eje is None: 
            eje = spectrum_image.axes_manager[-1].axis
        spectrum_image = spectrum_image.data
    elif isinstance(spectrum_image, np.ndarray):
        pass
    else:
        raise TypeError('The spectrum image has not been provided in the appropriate format.')
        
    while True:
        #VALORES DE UMAP:
        inp = input("Do you want to change the UMAP values? (y/n):")
        if inp.lower() == 'y':
            min_dis = float(input("Enter the value for minimum distance (min_dis):"))
            n_neigh = int(input("Enter the value for number of neighbors (n_neigh):"))
        else: 
            print('UMAP values: min_dis = {} and n_neigh = {}'.format(min_dis, n_neigh))
            
        #Representar HDBSCAN:
        inp = input("Do you want to evaluate HDBSCAN values? (y/n):")
        if inp.lower() == 'y':
            for i in range(1,8): #search over min_samples
                for j in [100,200,300,400,500,600,700,800,900]: #search over min_cluster_size
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=j,
                                                min_samples=i)
                    clusterer.fit(umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_)
                    outliers = np.count_nonzero(clusterer.labels_ == -1)
                    total_points = clusterer.labels_.size
                    print(i,j,len(np.unique(clusterer.labels_)),'Percentage of outliers: {:.2f} %'.format((outliers/total_points)*100))
        else:
            print("HDBSCAN values not evaluated.")
        
        ##-----------------------------HDBSCAN----------------------------------##
        min_samp = int(input("Enter the minimum value for samples (min_samples):"))
        min_clust = int(input("Enter the minimum value for cluster size (min_cluster_size):"))
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_clust,
                                    min_samples=min_samp)
        clusterer.fit(umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_)
        print('Cluster values:',np.unique(clusterer.labels_))
        ##-----------------------------HDBSCAN----------------------------------##

        if mask is not None:
            new_value = -2
            clustering = np.full_like(spectrum_image[:,:,0], new_value)
            clustering[mask] = clusterer.labels_
        else:
            clustering = clusterer.labels_.reshape(spectrum_image.shape[0],spectrum_image.shape[1])

        ##-----------------------------Plots------------------------------------##
        cmap_db, limis = get_cmap(clustering)
        labels_map = hv.Image(\
            xr.Dataset({'Labels':( ['y','x'],clustering.reshape(spectrum_image.shape[0],spectrum_image.shape[1]))},\
            coords = {'x':np.arange(spectrum_image.shape[1]),'y':np.arange(spectrum_image.shape[0])}),\
            kdims = ['x','y']).opts(xaxis = None,yaxis = None,colorbar = True,\
            tools = ['hover'],\
            toolbar = None,\
            invert_yaxis = True,
            aspect = 'equal',\
            frame_height = 800,\
            frame_width = 300,\
            cmap = cmap_db,\
            clim = limis,\
            title = 'HDBSCAN map')
        bokeh_figure = hv.render(labels_map)
        show(bokeh_figure)
        
        dictio_curva_mask = dict()
        mean_spec = []
        for i, el in enumerate(np.unique(clustering)):
            valores = np.mean(spectrum_image.reshape(-1,spectrum_image.shape[-1])[clustering.reshape((spectrum_image.shape[0]*spectrum_image.shape[1])) == el],axis=0)
            mean_spec.append(valores)
            dictio_curva_mask['Label_{}'.format(el)] = \
            hv.Curve(xr.Dataset({'Label_{}'.format(el):(['Eloss'],valores)},coords={'Eloss':np.arange(0,spectrum_image.shape[-1],1)})).opts(color = cmap_db[i])

        lay_mask = hv.NdOverlay(dictio_curva_mask)\
            .opts(frame_height=300,frame_width=650,bgcolor = 'black',legend_cols = False,\
            legend_position='right',show_grid=True,yaxis = None,xlabel = 'Energy Loss [eV]',\
            title = 'Centroids of HDBSCAN on the UMAP embedding')
        
        bokeh_figure = hv.render(lay_mask)
        show(bokeh_figure)
        
        assign = input("Do you want to assign HDBSCAN outliers to the found clusters? (y/n):")
        if assign.lower() == 'y':
            ## Assign the outliers to the closest cluster:
            outliers = np.where(clusterer.labels_ == -1)[0]
            for i in outliers:
                distances = pairwise_distances_argmin_min(umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_[i].reshape(1, -1), 
                                                          umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_[clusterer.labels_ != -1])
                closest_cluster = clusterer.labels_[np.where(clusterer.labels_ != -1)[0][distances[0]]]
                clusterer.labels_[i] = closest_cluster
            print('Outliers assigned:',np.unique(clusterer.labels_))
        
            if mask is not None:
                new_value = 0
                clustering = np.full_like(spectrum_image[:,:,0], new_value)
                clustering[mask] = clusterer.labels_ +1
            else:
                clustering = clusterer.labels_.reshape(spectrum_image.shape[0],spectrum_image.shape[1])

            cmap_db, limis = get_cmap(clustering)
            labels_map = hv.Image(\
                xr.Dataset({'Labels':( ['y','x'],clustering.reshape(spectrum_image.shape[0],spectrum_image.shape[1]))},\
                coords = {'x':np.arange(spectrum_image.shape[1]),'y':np.arange(spectrum_image.shape[0])}),\
                kdims = ['x','y']).opts(xaxis = None,yaxis = None,colorbar = True,\
                tools = ['hover'],\
                toolbar = None,\
                invert_yaxis = True,
                aspect = 'equal',\
                frame_height = 300,\
                frame_width = 600,\
                cmap = cmap_db,\
                clim = limis,\
                title = 'HDBSCAN map')      
            bokeh_figure = hv.render(labels_map)
            show(bokeh_figure)
        
            dictio_curva_mask = dict()
            mean_spec = []
            for i, el in enumerate(np.unique(clustering)):
                valores = np.mean(spectrum_image.reshape(-1,spectrum_image.shape[-1])[clustering.reshape((spectrum_image.shape[0]*spectrum_image.shape[1])) == el],axis=0)
                mean_spec.append(valores)
                dictio_curva_mask['Label_{}'.format(el)] = \
                hv.Curve(xr.Dataset({'Label_{}'.format(el):(['Eloss'],valores)},coords={'Eloss':np.arange(0,spectrum_image.shape[-1],1)})).opts(color = cmap_db[i])
        
            lay_mask = hv.NdOverlay(dictio_curva_mask)\
                .opts(frame_height=300,frame_width=650,bgcolor = 'black',legend_cols = False,\
                legend_position='right',show_grid=True,yaxis = None,xlabel = 'Energy Loss [eV]',\
                    title = 'Average TsXM for the clusters in the HDBSCAN clustering on the UMAP')
            
            bokeh_figure = hv.render(lay_mask)
            show(bokeh_figure)
            
            zers = np.zeros((umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_.shape[0],3))
            zers[:,:-1] = umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_
            zers[:,-1] = clusterer.labels_ #clusterer que no clustering
            
            if mask is not None:
                cmap_db = cmap_db[1:]
                
            embed_mask = hv.Points(zers,vdims = ['color'])\
            .opts(frame_width = 650,frame_height = 300,toolbar=None,fill_alpha = 0.5,bgcolor = 'black',\
                  line_alpha = 0,line_width = 0.15,size = 2.5,xaxis = None,yaxis = None,\
                  cmap = cmap_db,\
                  show_legend=True,\
                  color='color',\
                  shared_axes = False,\
                  title = 'UMAP embedding')
            
            bokeh_figure = hv.render(embed_mask)
            show(bokeh_figure)
        inp = input("Are you satisfied with the clustering results and want to finish? (y/n):")
        if inp.lower() == 'y':
            break

    #--------------------------------------------------------------------------------------------------------------#    
    cmap_db, _ = get_cmap(clustering)
    cmap_ = mcolors.ListedColormap(cmap_db)

    grouping_labels = np.array(clustering).reshape(spectrum_image.shape[0],spectrum_image.shape[1])
    a = np.unique(grouping_labels)
    #--------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------#
    labels_centros_norm = []
    labels_centros = []
    for i in range(0,len(a)):
        labels_centros_norm.append(normalize(spectrum_image[grouping_labels==a[i]].mean(axis=0).reshape(1,-1),norm='max',return_norm=True))
        labels_centros.append(spectrum_image[grouping_labels==a[i]].mean(axis=0))
    #--------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------------#
    zers = np.zeros((umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_.shape[0],3))
    zers[:,:-1] = umap_dict['udata_{}_{}'.format(min_dis,n_neigh)].embedding_
    zers[:,-1] = clusterer.labels_
    x = zers[:, 0]
    y = zers[:, 1]
    c = zers[:, 2]
    #--------------------------------------------------------------------------------------------------------------#    
    
    fig = plt.figure(dpi=dpi)
    bar = plt.imshow(grouping_labels,cmap=cmap_)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(bar,label='Label value',ticks=a)
    plt.tight_layout()
    
    fig1 = plt.figure(dpi=dpi)
    for i in range(0,len(labels_centros)):
        plt.plot(eje,labels_centros[i],label=str(a[i]),color = mcolors.to_rgba(cmap_db[i]),linewidth=0.8,linestyle=':')
    plt.xlabel('Energy axis')
    plt.ylabel('Intensity')
    plt.title('Centroids')
    plt.xlim(eje[0],eje[-1])
    plt.legend(fontsize=7)
    plt.tight_layout()
    
    fig2 = plt.figure(dpi=dpi)
    for i in range(0,len(labels_centros_norm)):
        plt.plot(eje,labels_centros_norm[i][0][0],label=str(a[i]),color = mcolors.to_rgba(cmap_db[i]),linewidth=0.8)
    plt.xlabel('Energy axis')
    plt.ylabel('Normalized Intensity')
    plt.title('Normalized Centroids')
    plt.xlim(eje[0],eje[-1])
    plt.ylim(-0.1,1.05)
    plt.legend(fontsize=7)
    plt.tight_layout()
    
    if mask is not None:
        cmap_db = cmap_db[1:]
        cmap_ = mcolors.ListedColormap(cmap_db)

    fig3 = plt.figure(dpi=dpi)
    plt.style.use("dark_background")
    plt.scatter(x, y, c=c, cmap=cmap_, alpha=0.5, s=.5)
    plt.title("UMAP embedding")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    plt.show()
    
    # Guardamos las imagenes:
    if filename is None: 
        inp = input("Do you want to save the figures? (y/n):")
        if inp.lower() == 'y':
            filename = input("Provide the name you want to save the images with:")
            if type(filename)==str:
                fig.savefig(filename+'_map')
                fig1.savefig(filename+'_centroids')
                fig2.savefig(filename+'_norm_centroids')
                fig3.savefig(filename+'_embbeding')
                print('Figuras guardadas')
                return grouping_labels
            else: 
                print('No valid name provided to save the figures,'+ \
                'returning them in case you want to save them later.')
                return (fig,fig1,fig2), grouping_labels
        else:
            print('Figures not saved')
            return (fig,fig1,fig2,fig3), grouping_labels
    else: 
        fig.savefig(filename+'_map')
        fig1.savefig(filename+'_centroids')
        fig2.savefig(filename+'_norm_centroids')
        fig3.savefig(filename+'_embbeding')
        print('Figuras guardadas')
        return grouping_labels
    
