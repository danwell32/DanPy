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

#Xarray:
import xarray as xr


#Sklearn: 
import sklearn
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

#Holoviews, Bokeh and panel:
import holoviews as hv
import panel as pn
import bokeh
from bokeh import palettes
from bokeh.io import export_svgs

#UMAP:
import umap

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
def get_cmap(labels, noise = 'grey',paleta = 'Spectral'):
    '''
    Devuelve un mapa de colores para representar los etiquetas resultantes de un algoritmo de clustering.

    Parameters:
        labels (np.array): Etiquetas resultantes de un modelo de clustering.
        noise (str): Color a utilizar para los puntos ruidosos. Valores permitidos: 'grey', 'magenta'.
        paleta (str): Nombre de la paleta de colores a utilizar como base.

    Returns:
        cmap_db (list): Lista de colores para utilizar en Holoviews en las etiquetas.
        limis (tuple): Límites para la barra de colores en la imagen de Holoviews para las etiquetas.
    '''
    if noise not in ['grey','magenta']: noise = 'grey'
    #The first thing, setting up the color for the noise datapoints
    if -2 in np.unique(labels):
        if noise == 'grey': cmap_db = ['lightgrey']
        else: cmap_db = ['magenta']
        #The rest of the colormap - notive we have -1 - noisy datapoints
        if np.unique(labels).size <= 12 and np.unique(labels).size >= 3:
            cmap_db.extend(list(palettes.all_palettes[paleta][np.unique(labels).size-1]))
        elif np.unique(labels).size == 2:
            cmap_db.append('dimgrey')
        elif np.unique(labels).size == 3:
            cmap_db.append('navy')
            cmap_db.append('limegreen')
        elif np.max(labels) >= 13:
            #It is unlikely that this would go beyond 256
            pal = list(palettes.all_palettes['Turbo'][256])
            if len(np.unique(labels)) > 256:
                raise Exception('Too many labels.. check if the clustering is well done')
            else: 
                nlabs = len(np.unique(labels))
                cmap_db = [el[0] for el in np.array_split(pal,nlabs)]
        else:
        # New version - much more stable
            cmap_db.append('indigo')
            cmap_db.extend(list(palettes.all_palettes[paleta][11]))
        #Lastthing - limits
        limis = (-1.5,len(cmap_db)-1.5)
    else:
        cmap_db = []
        if np.unique(labels).size <= 11 and np.unique(labels).size > 2:
            cmap_db.extend(list(palettes.all_palettes[paleta][np.unique(labels).size]))
        elif np.unique(labels).size == 2:
            cmap_db.extend(['lightgrey','dimgrey'])
        elif np.max(labels) >= 12:
            #It is unlikely that this would go beyond 256
            pal = list(palettes.all_palettes['Turbo'][256])
            if len(np.unique(labels)) > 256:
                raise Exception('Too many labels, check if the clustering is well done.')
            else:
                nlabs = len(np.unique(labels))
                cmap_db = [el[0] for el in np.array_split(pal,nlabs)]
        else:
            # New version - much more stable
            cmap_db.append('indigo')
            cmap_db.extend(list(palettes.all_palettes[paleta][11]))
        limis = (-0.5,len(cmap_db)-0.5)
    assert len(cmap_db) == np.unique(labels).size, print(len(cmap_db),np.unique(labels).size)
    return cmap_db,limis
#-------------------------------------------------------------------------------------------------------------------#
