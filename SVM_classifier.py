### Funciones de este modulo: 
# def kmeans_clustering(matrix, n_cluster, norma='l2')
# def train_and_test_svm_model(samples, labels, kernel, C=1., gamma=None, coef0=None, test_size=0.35, cv=10, probability=True)
# def reduce_energy_resolution(spectra, original_axis, new_axis)
# def identify_eels(spectra, model, probability=False, normalize=True, background=(620.,680.),crop=(615.,685.), back_point=(0,0), figure_name = None)
# def identify_eels_v2(spectra, model, probability=False, normalize=True, background=(620.,680.),crop=(615.,685.), back_point=(0,0), figure_name = None)
# def optimize_SVM_clasifiers(parameters,samples,labels,test_size=0.35,file_save=None)
# def displace_spectra(x, num_displacements, samples_per_displacement, max_displacement)
# def create_displaced_dataset(X, Y, num_displacements=100, samples_per_displacement=10, max_displacement=10)
# def classify_SI(spectra,model, ref_spectra, probability=True, normalize=True, background=None, crop=None,figure_name=None)
# def classify_SI_back(spectra, model, ref_spectra, probability=True, normalize=True, background=None, crop=None, back_point=(0,0),figure_name=None):

# Modulos importados: 
import numpy as np
from time import time
import copy as cp
import warnings
import joblib

#Sklearn: 
import sklearn
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, NMF

#Matplotlib: 
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as mcolors

#Hyperspy:
import hyperspy.api as hs

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
        Normalización que queremos aplicar. Opciones: ‘l1’, ‘l2’, ‘max’, None.
    Returns:
    --------
    labels: numpy array. (x,y)
        Matriz con las etiquetas de cada cluster. 
    centres: numpy array. (n_cluster,eloss)
        Matriz que contiene los centroides de cada cluster identificado (estos centroides provienen de los espectros normalizados). 
    '''
    allowed_norms = ['l1', 'l2', 'max', None]
    if norma not in allowed_norms:
        raise ValueError(f"La norma propuesta no es valida, debe ser alguna de estas: {allowed_norms}.")
        
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
def translation(inputs, translation):
    """
    Apply translations on 1D NumPy arrays.
    
    Parameters
    ----------
    inputs : array-like
        Input array to apply translation on.
    translation : int
        Translation value to apply.
        
    Returns
    -------
    ndarray
        Translated input array.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    trans = int(translation)
    n_features = inputs.shape[-1]

    val = np.random.randint(1, 5)
    if val % 2 == 0:  # Shift to the left.
        inputs_initial = inputs[:, trans:n_features]
        final = inputs[:, n_features-trans:n_features]
        return np.concatenate((inputs_initial, final), axis=1)
    else:  # Shift to the right.
        inputs_final = inputs[:, 0:n_features-trans]
        initial = inputs[:, 0:trans]
        return np.concatenate((initial, inputs_final), axis=1)
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def displace_spectra(x, num_displacements, samples_per_displacement, max_displacement):
    """
    Displace spectra by applying translations on the input array.
    
    Parameters
    ----------
    x : array-like
        Input array to displace.
    num_displacements : int
        Number of displacements to perform.
    samples_per_displacement : int
        Number of samples for each displacement.
    max_displacement : int
        Maximum displacement value.
        
    Returns
    -------
    ndarray
        Displaced input array.
    """
    x_dis = []
    for i in range(num_displacements):
        val = np.random.randint(1, max_displacement)
        idx = np.random.randint(0, x.shape[0], samples_per_displacement)
        x_dis.append(translation(x[idx], val))
    x_dis = np.array(x_dis).reshape(num_displacements * samples_per_displacement, x.shape[-1])
    return x_dis
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def create_displaced_dataset(X, Y, num_displacements=100, samples_per_displacement=10, max_displacement=10):
    """
    Create displaced spectra dataset by applying translations on the input arrays.
    
    Parameters
    ----------
    X : array-like
        Input array containing x data.
    Y : array-like
        Input array containing y data.
    num_displacements : int, optional
        Number of displacements to perform, by default 100.
    samples_per_displacement : int, optional
        Number of samples for each displacement, by default 10.
    max_displacement : int, optional
        Maximum displacement value, by default 10.
        
    Returns
    -------
    X_ : ndarray
        Concatenated array of original x data and displaced x data.
    labels_ : ndarray
        Concatenated array of original y data and displaced y data.
    """
    x = np.array(X)
    y = np.array(Y)

    unique_labels = np.unique(y)
    
    x_dis_list = []
    labels_dis_list = []
    
    for label in unique_labels:
        x_label = x[y == label]
        
        x_label_dis = displace_spectra(x_label, num_displacements, samples_per_displacement, max_displacement)
        x_dis_list.append(x_label_dis)
        
        labels_dis_list.append(np.full((num_displacements * samples_per_displacement,), label))

    x_dis = np.concatenate(x_dis_list, axis=0)
    labels_dis = np.concatenate(labels_dis_list, axis=0)

    X_ = np.concatenate((x, x_dis), axis=0)
    labels_ = np.concatenate((y, labels_dis), axis=0)
    return X_, labels_
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def train_and_test_svm_model(samples, labels, kernel, C=1., gamma=None, coef0=None, test_size=0.35, cv=10, probability=True):
    '''
    Entrena y evalúa un modelo de soft-margin SVM con los parámetros proporcionados.
    
    Parameters:
    -----------
    samples: numpy array.
        Conjunto de datos espectrales. Debe tener forma (n_spectra, n_features).
    labels: numpy array.
        Etiquetas del conjunto de datos espectrales. Debe tener forma (n_spectra).
    test_size: float, optional. (default=0.35)
        Proporción del conjunto de datos que se incluirá en la separación de prueba.
    cv: int, optional. (default=10)
        Validación cruzada en pliegues.
    kernel: string. 
        Tipo de kernel a usar en el estimador SVM. 
        Valores válidos: 'rbf', 'linear', 'sigmoid', 'sdg', 'cosine'.
    C: float, optional. (default=1.)
        Parámetro de regularización.
    gamma: float, optional. 
        Coeficiente de kernel para 'rbf' y 'sigmoid'.
    coef0: float, optional.
        Término independiente de la función kernel.
    probability: bool, optional. (default=True)
        Indica si se debe habilitar la capacidad de calcular probabilidades en el modelo SVM.
        
    Returns:
    --------
    model: objeto SVM entrenado.
    '''
    allowed_kernels = ['rbf', 'linear', 'sigmoid', 'sdg', 'cosine']
    if kernel not in allowed_kernels:
        raise ValueError(f"kernel debe ser uno de {allowed_kernels}")
    
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size)
    
    if kernel == 'rbf':
        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability, decision_function_shape='ovr', class_weight='balanced')    
    elif kernel == 'linear':
        model = SVC(C=C, kernel=kernel, probability=probability, decision_function_shape='ovr', class_weight='balanced')
    elif kernel == 'sigmoid':
        model = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, probability=probability, decision_function_shape='ovr', class_weight='balanced')
    elif kernel == 'sdg':
        model = SGDClassifier(class_weight='balanced')
    elif kernel == 'cosine': 
        model = SVC(C=C, kernel=cosine_similarity, probability=probability, decision_function_shape='ovr', class_weight='balanced')
    else: 
        model = SVC(C=C, kernel=kernel, probability=probability, decision_function_shape='ovr', class_weight='balanced')
    
    # Se realiza una validación cruzada sobre el conjunto de entrenamiento
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv)
    print(f"Scores de validación cruzada: {cv_scores}")
    print(f"Score promedio de validación cruzada: {np.mean(cv_scores)}")

    # Se entrena el modelo con el conjunto de entrenamiento
    model.fit(x_train, y_train)

    # Se hacen las predicciones con el conjunto de prueba
    y_pred = model.predict(x_test)

    # Se calcula la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión en conjunto de prueba: {accuracy}")

    return model
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def reduce_energy_resolution(spectra, original_axis, new_axis):
    """
    Reduce the energy resolution of the input spectra by interpolating the
    data onto a new axis.
    
    Parameters
    ----------
    spectra : array-like
        Input array containing spectra data.
    original_axis : array-like
        Original energy axis of the spectra.
    new_axis : array-like
        New energy axis onto which the spectra will be interpolated.
        
    Returns
    -------
    spectra_data : ndarray
        Array containing the spectra data with reduced energy resolution.
    """
    assert spectra.shape[-1] == len(original_axis)
    end_spectra = []
    spec_norm = np.array(spectra)
    for i in range(spec_norm.shape[0]):
        end_spectra.append(np.interp(x=new_axis, xp=original_axis, fp=spec_norm[i]))
    spectra_data = np.array(end_spectra)
    return spectra_data
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def optimize_SVM_clasifiers(parameters,samples,labels,test_size=0.35,file_save=None):
    '''
    This function optimizes the hyperparameters of a SVM estimator, and saves the model in a file.

    -parameters: dict or list of dictionaries. 
        Dictionary with parameters names (string) as keys and lists of parameter settings to try as values. 
        The parameters keys are: kernel, C range, gamma range, and r range.
        Examples of parameters: 
        Linear kernel: parameters = {"kernel":["linear"],"C":np.logspace(-3,4,8)}
        RBF kernel: parameters = {"kernel":["rbf"],"C":np.logspace(-3,4,8),"gamma":np.logspace(-6,4,11)}
        Sigmoid kernel: parameters = {"kernel":["sigmoid"],"C":np.logspace(-3,4,8),
                                    "gamma":np.logspace(-6,4,11),"coef0":np.logspace(-5,3,9)}
    -samples: array-like of shape (n_spectra,n_features)
        Spectral dataset.
    -labels: array-like of shape (n_spectra).
        Labels of the spectral dataset.
    test_size: float from 0 to 1. (default 0.35) 
        It represents the proportion of the dataset to include in the test split.
    fig_save: string. 
        Name of the file to save the plot. 

    Return:
    -------
    A dict of numpy ndarrays with the results of the Gridsearch routine.
    '''
    
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size)
    svm = SVC(probability=False,decision_function_shape='ovr',class_weight='balanced',kernel=cosine_similarity)
    grid_time = time()
    results = GridSearchCV(svm,parameters,cv=10,n_jobs=-1,verbose=10)
    results.fit(x_train,y_train)
    grid_time = time() - grid_time

    print('Time comsuming of the Gridsearch: ',grid_time,'seconds. \n')
    print('The best model is: \n')
    print(results.best_estimator_)

    if file_save is not None:
        joblib.dump(results, file_save)
        print('Gridsearch results saved.')
    else: 
        print('Gridsearch results are not saved.')

    return results
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
def identify_eels(spectra, model, ref_spectra, probability=True, normalize=True, background=(620.,680.),
                  crop=(615.,685.), back_point=(0,0), figure_name = None):
    """
    Classify a spectrum image using SVM or neural network. 
        - spectra: filename or hyperspy spectrum image
        - model: sklearn SVM model or neural network. 
        - probability: boolean. Whether to use probability or not.
        - normalize: boolean. Whether to normalize the spectrum data or not.
        - background: removebackground. Background energy range to be removed.
        - crop: crop the energy range. Energy range to be cropped.
        - back_point: background point. Point to identify the background.
        - figure_name: name of the figure to be saved. None by default.
    """
    #----------------------------------------------------------------------#
    # Los spectros de referencia para entrenar:
    # global spectra_data, y_total

    #----------------------------------------------------------------------#
    #Load by filename or hyperspy spectrum image. 
    if type(spectra) == str:
        spectrum_image = hs.load(spectra)
    else:
        spectrum_image = spectra
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    #Removebackground:
    if background == None:
        print('No remove background applied.')
    else:
        a, b = background
        spectrum_image = spectrum_image.remove_background((a,b))
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    #Crop de energy from the convenient range:
    if crop == None:
        print('No crop applied.')
    else: 
        c, d = crop
        spectrum_image = spectrum_image.isig[c:d]
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    #Pasa el spectro a numpy array:
    spectrum_data = spectrum_image.data
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # Normalization: 
    if normalize:
        spectrum_data_ = spectrum_data.copy()
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_, _ = sklearn.preprocessing.normalize(spectrum_data_,norm = 'max',return_norm=True) #normalize
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
    else:
        spectrum_data_ = spectrum_data.copy()
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    resol_old = spectrum_image.axes_manager["Energy loss"].scale
    old_len_Eaxis = len(spectrum_image.axes_manager[-1].axis)

    if crop == None: 
        new_len_Eaxis = ref_spectra.shape[-1]
        resol_new = 'Not assigned.'
    else:
        resol_new = (d-c)/ref_spectra.shape[-1]
        new_len_Eaxis = old_len_Eaxis
    
    print(resol_new,resol_old,old_len_Eaxis,new_len_Eaxis)

    if resol_old != resol_new and old_len_Eaxis != new_len_Eaxis:
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_ = reduc_energy_resol(spectrum_data_,c,d,resol_new,resol_old)
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data_.shape[-1])) #reshape
    else: 
        pass
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    #Plot the spectra:
    plt.figure(dpi=150)
    plt.title('Comparative of training spectra and SI spectra')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(np.arange(0,ref_spectra.shape[-1],1),ref_spectra[idx],linewidth=0.5,color='r')#,label='Mn2')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(np.arange(0,ref_spectra.shape[-1],1),ref_spectra[idx],linewidth=0.5,color='b')#,label='Mn3')
    #idx = np.random.randint(0,spectra_data.shape[0])
    #plt.plot(np.arange(c,d,resol_new),ref_spectra[300],linewidth=0.5,color='y',label='Mn4')
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(np.arange(0,spectrum_data_.shape[-1],1),spectrum_data_[idx][idy],linewidth=0.5,color='fuchsia',label='x: {}, y: {}'.format(idx,idy))
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(np.arange(0,spectrum_data_.shape[-1],1),spectrum_data_[idx][idy],linewidth=0.5,color='purple',label='x: {}, y: {}'.format(idx,idy))
    plt.legend()
    plt.show()
    #----------------------------------------------------------------------# 

    #----------------------------------------------------------------------#
    #Clustering para eliminar el background: 
    labels, _ = kmeans_clustering(spectrum_data,n_cluster=2,norma=None)
    
    #Comprovar que el clustering funcione bien: 
    _, cuentas = np.unique(labels,return_counts=True)
    for i in cuentas: 
        if i < (labels.shape[0]*labels.shape[1])*0.05:
            raise ValueError("Uno o mas clusteres tienen menos del 5% del total de los puntos de la imagen.")
        else: 
            pass
        
    #Vamos a comparar con el punto de fondo: 
    x_back, y_back = back_point
    #value es el indicador del label del background
    if labels[x_back][y_back] == 0:
        value = 1
    else:
        value = 0
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    num_classes = model.classes_.shape[0]
    
    #----------------------------------------------------------------------#
    cmap_original = plt.get_cmap("plasma_r")
    custom_cmap = cmap_original.copy()
    custom_cmap.set_under('black')
    custom_norm = mcolors.Normalize(vmin=0, vmax=num_classes)
    custom_norm1 = mcolors.Normalize(vmin=0, vmax=1)
    #----------------------------------------------------------------------#


    #Binary classification:         
    labels_1 = labels.copy()
    spectrum_labeled = spectrum_data_[labels==value]
    clasificacion_1 = model.predict(spectrum_labeled)
    count = 0
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels_1[x][y] == value:
                labels_1[x][y] = clasificacion_1[count]
                count+=1
            else: 
                labels_1[x][y] = -1
    
    if probability:
        #Probabilistic classification: 
        spectrum_labeled = spectrum_data_.reshape(spectrum_data_.shape[0]*spectrum_data_.shape[1],spectrum_data_.shape[-1])
        clasificacion_0 = model.predict_proba(spectrum_labeled)
        clasificacion_0 = clasificacion_0.reshape(spectrum_data_.shape[0],spectrum_data_.shape[1],clasificacion_0.shape[-1])
        
        # Combine the probability values into a single RGB image
        probs = clasificacion_0
        combined_map = np.zeros((probs.shape[0],probs.shape[1],probs.shape[-1]), dtype=np.float64)
        for i in range(0,clasificacion_0.shape[-1]):
            combined_map[..., i] = probs[..., i]
    
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                if labels[x][y] == value:
                    pass
                else:
                    combined_map[x,y,:] = -0.99
        #----------------------------------------------------------------------#

        dim = clasificacion_0.shape[-1]
        fig, axs = plt.subplots(int(dim/2+1), 2, figsize=(10, 10), dpi=200)
        # Add scale bar to all subplots
        scale_length = spectrum_image.axes_manager['x'].scale
        scale_unit = spectrum_image.axes_manager['x'].units

        im = axs[0, 0].imshow(labels_1, cmap=custom_cmap, norm=custom_norm)
        axs[0, 0].set_title('Binary clasification')
        fig.colorbar(im, ax=axs[0, 0], cmap=custom_cmap, label='Class')
        sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location='lower left', 
                      length_fraction=0.4, height_fraction=0.5, width_fraction=0.015, scale_loc='top', 
                      font_properties={'size':14})
        axs[0, 0].add_artist(sb)
        
        contador = 0
        for i in range(0,clasificacion_0.shape[-1]):
            if i%2==0:
                axs[contador, 1].set_title('Probabilistic clasification')
                im = axs[contador, 1].imshow(combined_map[..., i], cmap=custom_cmap, norm=custom_norm1)
                axs[contador, 1].set_title('Label {}'.format(i))
                fig.colorbar(im, ax=axs[contador, 1], cmap=custom_cmap, label='Class')
                sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location='lower left', 
                          length_fraction=0.4, height_fraction=0.5, width_fraction=0.015, scale_loc='top', 
                          font_properties={'size':14})
                axs[contador, 1].add_artist(sb)
            else:
                contador += 1
                axs[contador, 0].set_title('Probabilistic clasification')
                im = axs[contador, 0].imshow(combined_map[..., i], cmap=custom_cmap,norm=custom_norm1)
                axs[contador, 0].set_title('Label {}'.format(i))
                fig.colorbar(im, ax=axs[contador, 0], cmap=custom_cmap, label='Oxidation state')
                sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location='lower left', 
                          length_fraction=0.4, height_fraction=0.5, width_fraction=0.015, scale_loc='top', 
                          font_properties={'size':14})
                axs[contador, 0].add_artist(sb)
        # Remove tick labels
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    else:
        #------------------Binary clasification----------------------------#        
        labels_1 = labels.copy()
        spectrum_labeled = spectrum_data_[labels==value]
        clasificacion_1 = model.predict(spectrum_labeled)
        count = 0
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                if labels_1[x][y] == value:
                    labels_1[x][y] = clasificacion_1[count]
                    count+=1
                else: 
                    labels_1[x][y] = 0
    
        fig2 = plt.figure(figsize=(8, 8), dpi=200)
        ax = plt.axes()
        im = plt.imshow(labels_1, cmap=custom_cmap, norm=custom_norm)
        plt.title('Binary clasification')
        fig2.colorbar(im, cmap=custom_cmap, label='Classes')
        sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location='lower left', 
                      length_fraction=0.4, height_fraction=0.5, width_fraction=0.015, scale_loc='top', 
                      font_properties={'size':14})
        ax.add_artist(sb)             
        plt.show()
        
    if figure_name !=None:
        if probability:
            fig.savefig(figure_name + '_probs',dpi=500)
        else:
            fig2.savefig(figure_name + '_binary',dpi=500)
        
    return None
#-------------------------------------------------------------------------------------------------------------------#

#Modificacion de identify_eels que incorpora la suma de probabilidades y plotea este plot con las probabilidades sumadas en lugar de ser binario: 
#-------------------------------------------------------------------------------------------------------------------#
def identify_eels_v2(spectra, model, ref_spectra, probability=False, normalize=True, background=None, \
                  crop=None, back_point=(0,0), figure_name = None):
    """
    Clasificar por SVM o neural network una spectrum image. 
        - spectra: filename or hyperspy spectrum image
        - model: sklearn SVM model or neural network. 
        - probability: boolean.
        - normalize: boolean.
        - background: removebackground.
        - crop: crop the energy range. 
        - back_point: punto del background. 
    """
    
    #Comprobamos si es una array o una nombre. 
    if type(spectra) == str:
        spectrum_image = hs.load(spectra)
    else:   
        spectrum_image = spectra
    
    #Removebackground:
    if background == None:
        print('No remove background applied.')
    else:
        a, b = background
        spectrum_image = spectrum_image.remove_background((a,b))
    
    #Crop de energy from the convenient range:
    if crop == None:
        print('No crop applied.')
    else: 
        c, d = crop
        spectrum_image = spectrum_image.isig[c:d]
    
    #Pasa el spectro a numpy array:
    spectrum_data = spectrum_image.data
    
    #-----------------------------------------------------------------------------------#
    #Comprovamos que las dimensiones de entrada del modelo y de los datos facilitados sean correctos.
    if spectrum_data.shape[-1] < ref_spectra.shape[-1]:
        raise ValueError("The dimensions of the spectrum of the provided data do not match those of the training data. Please check the model.")
    #-----------------------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    # Normalization: 
    if normalize:
        spectrum_data_ = spectrum_data.copy()
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_, _ = sklearn.preprocessing.normalize(spectrum_data_,norm = 'max',return_norm=True) #normalize
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
    else:
        spectrum_data_ = spectrum_data.copy()
    #----------------------------------------------------------------------#
    
    #----------------------------------------------------------------------#
    resol_old = spectrum_image.axes_manager["Energy loss"].scale
    old_len_Eaxis = len(spectrum_image.axes_manager[-1].axis)
    #----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    if crop == None: 
        new_len_Eaxis = ref_spectra.shape[-1]
        resol_new = 'Not assigned.'
    else:
        resol_new = (d-c)/ref_spectra.shape[-1]
        new_len_Eaxis = old_len_Eaxis

    print(resol_new,resol_old,old_len_Eaxis,new_len_Eaxis)
    #----------------------------------------------------------------------#
    #if resol_old != resol_new and old_len_Eaxis != new_len_Eaxis:
    #    spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
    #    # Llamar a la función adapt_energy_axis para adaptar los ejes de energía
    #    ref_energy_axis = np.linspace(c, d, new_len_Eaxis)
    #    old_energy_axis = np.linspace(c, d, old_len_Eaxis)
    #    spectrum_data_ = adapt_energy_axis(spectrum_data_, ref_energy_axis, old_energy_axis)
    #    spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data_.shape[-1])) #reshape
    #else: 
    #    pass
    #----------------------------------------------------------------------#
    
    if resol_old != resol_new and old_len_Eaxis != new_len_Eaxis:
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_ = reduc_energy_resol(spectrum_data_,c,d,resol_new,resol_old)
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data_.shape[-1])) #reshape
    else: 
        pass
    
    #----------------------------------------------------------------------#
    #Plot the spectra:
    plt.figure(dpi=150)
    plt.title('Comparative of training spectra and SI spectra')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(np.arange(0,ref_spectra.shape[-1],1),ref_spectra[idx],linewidth=0.5,color='red')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(np.arange(0,ref_spectra.shape[-1],1),ref_spectra[idx],linewidth=0.5,color='blue')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(np.arange(0,ref_spectra.shape[-1],1),ref_spectra[idx],linewidth=0.5,color='green')
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(np.arange(0,spectrum_data_.shape[-1],1),spectrum_data_[idx][idy],linewidth=0.5,color='fuchsia',label='x: {}, y: {}'.format(idx,idy))
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(np.arange(0,spectrum_data_.shape[-1],1),spectrum_data_[idx][idy],linewidth=0.5,color='purple',label='x: {}, y: {}'.format(idx,idy))
    plt.legend()
    plt.show()
    #----------------------------------------------------------------------# 
    
    #------------------Clustering for background-------------------------#    
    labels, etiquetas = kmeans_clustering(spectrum_data,n_cluster=2,norma=None)
    
    #Comprovar que el clustering funcione bien: 
    _, cuentas = np.unique(labels,return_counts=True)
    for i in cuentas: 
        if i < (labels.shape[0]*labels.shape[1])*0.05:
            plt.figure(dpi=100)
            plt.title('Clustering for background')
            plt.imshow(labels)
            plt.show()
            plt.figure(dpi=100)
            plt.title('Clustering for centroids')
            plt.plot(etiquetas[0])
            plt.plot(etiquetas[1])
            plt.show()    
            raise ValueError("The clustering has not been able to separate the background from the signal.")
        else: 
            pass
    
    #Vamos a comparar con el punto de fondo: 
    x_back, y_back = back_point
    #value es el indicador del label del background
    if labels[x_back][y_back] == 0:
        value = 1
    else:
        value = 0
        
    #-----------------------------------------------------------------------------------#
    spectrum_labeled = spectrum_data_.reshape(spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data_.shape[-1])
    #----------------------------Clasificacion------------------------------------------#
    # Clasificador probabilistico:
    if probability:
        clasificacion_0 = model.predict_proba(spectrum_labeled)
    # Clasificador binario: 
    clasificacion_1 = model.predict(spectrum_labeled)
    num_classes = model.classes_.shape[0]
    
    #-----------------------------------------------------------------------------------#
    if probability: 
        clasificacion_0 = clasificacion_0.reshape(spectrum_data.shape[0],spectrum_data.shape[1],clasificacion_0.shape[-1])        
        probs = clasificacion_0
        # Calcular el valor de las classes ponderado
        classes_value = np.sum([(i + 1)*clasificacion_0[..., i] for i in range(num_classes)], axis=0)
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                if labels[x][y] == value:
                    pass
                else: 
                    classes_value[x, y] = 0
        #----------------------------------------------------------------------#
        cmap_original = plt.get_cmap("plasma_r")
        custom_cmap = cmap_original.copy()
        # Define el color negro para valores por debajo del limite inferior
        custom_cmap.set_under('black')
        custom_norm = mcolors.Normalize(vmin=1, vmax=num_classes)
        #----------------------------------------------------------------------#
        fig = plt.figure(dpi=200)
        im = plt.imshow(classes_value, cmap=custom_cmap, norm=custom_norm)
        cbar = plt.colorbar(im, cmap=custom_cmap, label='Class', norm=custom_norm)
        plt.title('Probabilistic classifier')
        cbar.set_ticks(np.arange(1,num_classes+1))
        cbar.set_ticklabels(np.arange(1,num_classes+1))
        plt.xticks([])
        plt.yticks([])
        scale_length = spectrum_image.axes_manager['x'].scale
        scale_unit = spectrum_image.axes_manager['x'].units
        sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location='lower left',length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
        plt.gca().add_artist(sb)
    
    #------------------Binary clasification----------------------------#        
    labels_1 = labels.copy()
    spectrum_labeled = spectrum_data_[labels==value]
    clasificacion_1 = model.predict(spectrum_labeled)
    count = 0
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels_1[x][y] == value:
                labels_1[x][y] = clasificacion_1[count]
                count+=1
            else: 
                labels_1[x][y] = 0
    #-----------------------------------------------------------------------------------#        

    #-----------------------------------------------------------------------------------#        
    fig2 = plt.figure(dpi=200)
    im = plt.imshow(labels_1, cmap=custom_cmap, norm=custom_norm)
    cbar = plt.colorbar(im,cmap=custom_cmap,label='Classes')
    plt.title('Binary classifier')
    cbar.set_ticks(np.arange(1,num_classes+1))
    cbar.set_ticklabels(np.arange(1,num_classes+1))
    plt.xticks([])
    plt.yticks([])
    scale_length = spectrum_image.axes_manager['x'].scale
    scale_unit = spectrum_image.axes_manager['x'].units
    sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location='lower left',length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
    plt.gca().add_artist(sb)
    
    #-----------------------------------------------------------------------------------#
    if figure_name !=None:
        if probability:
            fig.savefig(figure_name + '_probs',dpi=500)
        fig2.savefig(figure_name + '_binary',dpi=500)
        
    return None
#-------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
def classify_spectrum(spectra, 
                      model, 
                      ref_spectra, 
                      probability=True, 
                      normalize=True, 
                      background=None, 
                      crop=None, 
                      nn=False, 
                      figure_name=None):
    """
    Classify a spectrum image without background (all the points contain sample) using SVM.
    It will return the show the plots with the classification results. 
    
    Parameters:
    - spectra: filename or EELS hyperspy spectrum image
    - model: sklearn SVM model or neural network
    - ref_spectra: reference spectra
    - probability: boolean, optional (default=True)
    - normalize: boolean, optional (default=True)
    - background: tuple, optional (default=None)
    - crop: tuple, optional (default=None)
    - nn: boolean, optional (default=False), indicates if your model is a ANN from Tensorflow.
    - figure_name: string, optional (default=None)

    Returns:
    - None
    """

    # Check if the input is a file name or a hyperspy spectrum image
    if type(spectra) == str:
        spectrum_image = hs.load(spectra)
    elif type(spectra) == hs.signals.EELSSpectrum:
        spectrum_image = spectra
    else:
        raise ValueError("No valid data provided.")

    # Remove background
    if background is not None:
        a, b = background
        spectrum_image = spectrum_image.remove_background((a, b))

    # Crop the energy range
    if crop == None:
        c = spectrum_image.axes_manager[-1].axis[0]
        d = spectrum_image.axes_manager[-1].axis[-1]
        print('No crop applied.')
    else: 
        c, d = crop
        spectrum_image = spectrum_image.isig[c:d]

    # Convert the spectrum to a numpy array
    spectrum_data = spectrum_image.data

    # Normalize the data if requested
    if normalize:
        spectrum_data_ = spectrum_data.copy()
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_, _ = sklearn.preprocessing.normalize(spectrum_data_,norm = 'max',return_norm=True) #normalize
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
    else:
        spectrum_data_ = spectrum_data.copy()
        
    # Check if the dimensions of the input spectrum match the training data dimensions
    if spectrum_data.shape[-1] < ref_spectra.shape[-1]:
        print(f"The dimensions of the SI provided ({spectrum_data.shape[-1]}) is smaller than the training data ({ref_spectra.shape[-1]}).")
        print("It is recommended to reduce the training data resolution and not the evaluated Spectrum Image.")
    
    #----------------------------------------------------------------------#    
    #eje de energias de los espectros facilitados:
    Eaxis_si = spectrum_image.axes_manager[-1].axis
    #eje de energias de los datos de referencia: 
    Eaxis_ref = np.arange(c,d,(d-c)/ref_spectra.shape[-1])
    
    if Eaxis_si.shape[-1] != Eaxis_ref.shape[-1]:
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_ = reduce_energy_resolution(spectrum_data_, original_axis=Eaxis_si, new_axis=Eaxis_ref)
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data_.shape[-1])) #reshape
    else: 
        pass
    
    #----------------------------------------------------------------------#
    #Plot the spectra:
    plt.figure(dpi=150)
    plt.title('Comparative of training spectra and SI spectra')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(Eaxis_ref,ref_spectra[idx],linewidth=0.5,color='red')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(Eaxis_ref,ref_spectra[idx],linewidth=0.5,color='blue')
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(Eaxis_ref,spectrum_data_[idx][idy],linewidth=0.5,color='fuchsia',label='x: {}, y: {}'.format(idx,idy))
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(Eaxis_ref,spectrum_data_[idx][idy],linewidth=0.5,color='purple',label='x: {}, y: {}'.format(idx,idy))
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------#
    spectrum_labeled = spectrum_data_.reshape(spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data_.shape[-1])
    #-----------------------------------------------------------------------------------#    
    
    #----------------------------Clasificacion------------------------------------------#
    # Clasificador probabilistico:
    if probability:
        clasificacion_0 = model.predict_proba(spectrum_labeled)
    # Clasificador binario: 
    clasificacion_1 = model.predict(spectrum_labeled)
    if nn:
        num_classes = clasificacion_1.shape[-1]
        print(num_classes)
    else:
        num_classes = model.classes_.shape[0]
    #-----------------------------------------------------------------------------------#
    
    #---------------------------------Color_map-----------------------------------------#
    cmap_original = plt.get_cmap("plasma_r")
    custom_cmap = cmap_original.copy()
    # Define el color negro para valores por debajo del limite inferior
    custom_cmap.set_under('black')
    custom_norm = mcolors.Normalize(vmin=-0.01, vmax=num_classes-1)
    custom_norm_ = mcolors.Normalize(vmin=-0.01, vmax=1.)
    #-----------------------------------------------------------------------------------#
    fig_ = []
    #-----------------------------------------------------------------------------------#
    if probability: 
        clasificacion_0 = clasificacion_0.reshape(spectrum_data.shape[0],spectrum_data.shape[1],clasificacion_0.shape[-1])
        for i in range(num_classes):
        # Calcular el valor de las classes ponderado
            fig = plt.figure(dpi=200)
            fig_.append(fig)
            im = plt.imshow(clasificacion_0[:,:,i], cmap=custom_cmap, norm=custom_norm_)
            cbar = plt.colorbar(im, label='Probability')
            plt.title('Class_{}'.format(i))
            cbar.set_ticks(np.arange(0,1.05,0.25))
            cbar.set_ticklabels(np.arange(0,1.05,0.25))
            plt.xticks([])
            plt.yticks([])
            scale_length = spectrum_image.axes_manager['x'].scale
            scale_unit = spectrum_image.axes_manager['x'].units
            #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
            #plt.gca().add_artist(sb)
        classes_value = np.sum([(i + 2)*clasificacion_0[..., i] for i in range(num_classes)], axis=0)
        custom_norm_ox = mcolors.Normalize(vmin=2, vmax=3)
        # Calcular el valor de las classes ponderado
        fig = plt.figure(dpi=200)
        fig_.append(fig)
        im = plt.imshow(classes_value, cmap=custom_cmap, norm=custom_norm_ox)
        cbar = plt.colorbar(im, label='Oxidation State')
        #plt.title('Class_{}'.format(i))
        cbar.set_ticks(np.arange(2.,3.5,1.))
        cbar.set_ticklabels(np.arange(2,3.5,1))
        plt.xticks([])
        plt.yticks([])
        scale_length = spectrum_image.axes_manager['x'].scale
        scale_unit = spectrum_image.axes_manager['x'].units
        #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
        #plt.gca().add_artist(sb)
    #-----------------------------------------------------------------------------------#
    
    #----------------------------Binary clasification-----------------------------------#
    if nn:
        clasificacion_1 = tf.argmax(clasificacion_1,axis=-1)
        clasificacion_1 = clasificacion_1.numpy().reshape(spectrum_data.shape[0],spectrum_data.shape[1])#,num_classes)
    else:
        clasificacion_1 = clasificacion_1.reshape(spectrum_data.shape[0],spectrum_data.shape[1])
    #-----------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------#        
    fig1 = plt.figure(dpi=200)
    im = plt.imshow(clasificacion_1, cmap=custom_cmap, norm=custom_norm)
    cbar = plt.colorbar(im,label='Classes')
    plt.title('Binary classifier')
    cbar.set_ticks(np.arange(0,num_classes))
    cbar.set_ticklabels(np.arange(0,num_classes))
    plt.xticks([])
    plt.yticks([])
    scale_length = spectrum_image.axes_manager['x'].scale
    scale_unit = spectrum_image.axes_manager['x'].units
    #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
    #plt.gca().add_artist(sb)
    #-----------------------------------------------------------------------------------#   
    
    #-----------------------------------------------------------------------------------#
    #num_classes = model.classes_.shape[0]
    mean_spectra = np.zeros((num_classes, spectrum_data.shape[-1]))

    for i in range(num_classes):
        indices = np.where(clasificacion_1 == i)
        class_spectra = spectrum_data[indices[0], indices[1]]
        mean_spectra[i] = np.mean(class_spectra, axis=0)
    #-----------------------------------------------------------------------------------#  
    fig2 = plt.figure(dpi=300)
    for i in range(0,num_classes):
        plt.plot(Eaxis_ref,mean_spectra[i],label='{}'.format(i),linewidth=0.5,alpha=0.8)
    #plt.xlim(690.,750.)
    plt.title('Average spectrum of each class')
    plt.legend(title='Class') 
    
    #-----------------------------------------------------------------------------------#
    if figure_name is not None:
        if probability:
            cont = 0
            for figure in fig_:
                figure.savefig(figure_name + '_probs{}'.format(cont),dpi=500)
                cont += 1
        fig1.savefig(figure_name + '_binary',dpi=500)
        fig2.savefig(figure_name + '_centroids',dpi=500)
    #----------------------------------------------------------------------------------------------------------------#
    return None
#----------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
def classify_SI(spectra, 
                model, 
                ref_spectra, 
                probability=True, 
                normalize=True, 
                background=None, 
                crop=None,
                figure_name=None):
    """
    Classify a spectrum image without background (all the points contain sample) using SVM.
    It will return the show the plots with the classification results. 

    Parameters:
    - spectra: filename or EELS hyperspy spectrum image
    - model: sklearn SVM model or neural network
    - ref_spectra: reference spectra
    - probability: boolean, optional (default=True)
    - normalize: boolean, optional (default=True)
    - background: tuple, optional (default=None)
    - crop: tuple, optional (default=None)
    - nn: boolean, optional (default=False), indicates if your model is a ANN from Tensorflow.
    - figure_name: string, optional (default=None)

    Returns:
    - None
    """

    # Check if the input is a file name or a hyperspy spectrum image
    if type(spectra) == str:
        spectrum_image = hs.load(spectra)
    elif type(spectra) == hs.signals.EELSSpectrum:
        spectrum_image = spectra
    else:
        raise ValueError("No valid data provided.")

    # Remove background
    if background is not None:
        a, b = background
        spectrum_image = spectrum_image.remove_background((a, b))

    # Crop the energy range
    if crop == None:
        c = spectrum_image.axes_manager[-1].axis[0]
        d = spectrum_image.axes_manager[-1].axis[-1]
        print('No crop applied.')
    else: 
        c, d = crop
        spectrum_image = spectrum_image.isig[c:d]

    # Convert the spectrum to a numpy array
    spectrum_data = spectrum_image.data

    # Normalize the data if requested
    if normalize:
        spectrum_data_ = spectrum_data.copy()
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_, _ = sklearn.preprocessing.normalize(spectrum_data_,norm = 'max',return_norm=True) #normalize
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
    else:
        spectrum_data_ = spectrum_data.copy()
        
    # Check if the dimensions of the input spectrum match the training data dimensions
    if spectrum_data.shape[-1] < ref_spectra.shape[-1]:
        print(f"The dimensions of the SI provided ({spectrum_data.shape[-1]}) is smaller than the training data ({ref_spectra.shape[-1]}).")
        print("It is recommended to reduce the training data resolution and not the evaluated Spectrum Image.")
    
    #----------------------------------------------------------------------#    
    #eje de energias de los espectros facilitados:
    Eaxis_si = spectrum_image.axes_manager[-1].axis
    #eje de energias de los datos de referencia: 
    Eaxis_ref = np.arange(c,d,(d-c)/ref_spectra.shape[-1])
    
    if Eaxis_si.shape[-1] != Eaxis_ref.shape[-1]:
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_ = reduce_energy_resolution(spectrum_data_,eje_original=Eaxis_si,eje_nuevo=Eaxis_ref)
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data_.shape[-1])) #reshape
    else: 
        pass
    
    #----------------------------------------------------------------------#
    #Plot the spectra:
    plt.figure(dpi=150)
    plt.title('Comparative of training spectra and SI spectra')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(Eaxis_ref,ref_spectra[idx],linewidth=0.5,color='red')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(Eaxis_ref,ref_spectra[idx],linewidth=0.5,color='blue')
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(Eaxis_ref,spectrum_data_[idx][idy],linewidth=0.5,color='fuchsia',label='x: {}, y: {}'.format(idx,idy))
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(Eaxis_ref,spectrum_data_[idx][idy],linewidth=0.5,color='purple',label='x: {}, y: {}'.format(idx,idy))
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------#
    spectrum_labeled = spectrum_data_.reshape(spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data_.shape[-1])
    #-----------------------------------------------------------------------------------#    
    
    #----------------------------Clasificacion------------------------------------------#
    # Clasificador probabilistico:
    if probability:
        clasificacion_0 = model.predict_proba(spectrum_labeled)
    # Clasificador binario: 
    clasificacion_1 = model.predict(spectrum_labeled)
    num_classes = model.classes_.shape[0]
    #-----------------------------------------------------------------------------------#
    
    #---------------------------------Color_map-----------------------------------------#
    cmap_original = plt.get_cmap("plasma_r")
    custom_cmap = cmap_original.copy()
    # Define el color negro para valores por debajo del limite inferior
    custom_cmap.set_under('black')
    custom_norm = mcolors.Normalize(vmin=-0.01, vmax=num_classes-1)
    custom_norm_ = mcolors.Normalize(vmin=-0.01, vmax=1.)
    #-----------------------------------------------------------------------------------#
    fig_ = []
    #-----------------------------------------------------------------------------------#
    if probability: 
        clasificacion_0 = clasificacion_0.reshape(spectrum_data.shape[0],spectrum_data.shape[1],clasificacion_0.shape[-1])
        for i in range(num_classes):
        # Calcular el valor de las classes ponderado
            fig = plt.figure(dpi=200)
            fig_.append(fig)
            im = plt.imshow(clasificacion_0[:,:,i], cmap=custom_cmap, norm=custom_norm_)
            cbar = plt.colorbar(im, label='Probability')
            plt.title('Class_{}'.format(i))
            cbar.set_ticks(np.arange(0,1.05,0.25))
            cbar.set_ticklabels(np.arange(0,1.05,0.25))
            plt.xticks([])
            plt.yticks([])
            scale_length = spectrum_image.axes_manager['x'].scale
            scale_unit = spectrum_image.axes_manager['x'].units
            #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
            #plt.gca().add_artist(sb)
        """
        #Code to plot the oxidation state: (have to be adapted to the oxide)
        classes_value = np.sum([(i + 2)*clasificacion_0[..., i] for i in range(num_classes)], axis=0)
        custom_norm_ox = mcolors.Normalize(vmin=2, vmax=3)
        # Calcular el valor de las classes ponderado
        fig = plt.figure(dpi=200)
        fig_.append(fig)
        im = plt.imshow(classes_value, cmap=custom_cmap, norm=custom_norm_ox)
        cbar = plt.colorbar(im, label='Oxidation State')
        #plt.title('Class_{}'.format(i))
        cbar.set_ticks(np.arange(2.,3.5,1.))
        cbar.set_ticklabels(np.arange(2,3.5,1))
        plt.xticks([])
        plt.yticks([])
        scale_length = spectrum_image.axes_manager['x'].scale
        scale_unit = spectrum_image.axes_manager['x'].units
        #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
        #plt.gca().add_artist(sb)
        """
    #-----------------------------------------------------------------------------------#
    
    #----------------------------Binary clasification-----------------------------------#
    clasificacion_1 = clasificacion_1.reshape(spectrum_data.shape[0],spectrum_data.shape[1])
    #-----------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------#        
    fig1 = plt.figure(dpi=200)
    im = plt.imshow(clasificacion_1, cmap=custom_cmap, norm=custom_norm)
    cbar = plt.colorbar(im,label='Classes')
    plt.title('Binary classifier')
    cbar.set_ticks(np.arange(0,num_classes))
    cbar.set_ticklabels(np.arange(0,num_classes))
    plt.xticks([])
    plt.yticks([])
    scale_length = spectrum_image.axes_manager['x'].scale
    scale_unit = spectrum_image.axes_manager['x'].units
    #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
    #plt.gca().add_artist(sb)
    #-----------------------------------------------------------------------------------#   
    
    #-----------------------------------------------------------------------------------#
    num_classes = model.classes_.shape[0]
    mean_spectra = np.zeros((num_classes, spectrum_data.shape[-1]))

    for i in range(num_classes):
        indices = np.where(clasificacion_1 == i)
        class_spectra = spectrum_data[indices[0], indices[1]]
        mean_spectra[i] = np.mean(class_spectra, axis=0)
    #-----------------------------------------------------------------------------------#  
    fig2 = plt.figure(dpi=300)
    for i in range(0,num_classes):
        plt.plot(Eaxis_ref,mean_spectra[i],label='{}'.format(i),linewidth=0.5,alpha=0.8)
    #plt.xlim(690.,750.)
    plt.title('Average spectrum of each class')
    plt.legend(title='Class') 
    
    #-----------------------------------------------------------------------------------#
    if figure_name is not None:
        if probability:
            cont = 0
            for figure in fig_:
                figure.savefig(figure_name + '_probs{}'.format(cont),dpi=500)
                cont += 1
        fig1.savefig(figure_name + '_binary',dpi=500)
        fig2.savefig(figure_name + '_centroids',dpi=500)
    #----------------------------------------------------------------------------------------------------------------#
    return None
#----------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------#
def classify_SI_back(spectra, 
                     model, 
                     ref_spectra, 
                     probability=True, 
                     normalize=True, 
                     background=None, 
                     crop=None, 
                     back_point=(0,0),
                     figure_name=None):
    """
    Classify a spectrum image with background (no sample region) using SVM or ANN.
    It will return the show the plots with the classification results. 

    Parameters:
    - spectra: filename or EELS hyperspy spectrum image
    - model: sklearn SVM model or neural network
    - ref_spectra: reference spectra
    - probability: boolean, optional (default=True)
    - normalize: boolean, optional (default=True)
    - background: tuple, optional (default=None)
    - crop: tuple, optional (default=None)
    - back_point: point of the background. 
    - figure_name: string, optional (default=None)

    Returns:
    - None
    """

    # Check if the input is a file name or a hyperspy spectrum image
    if type(spectra) == str:
        spectrum_image = hs.load(spectra)
    elif type(spectra) == hs.signals.EELSSpectrum:
        spectrum_image = spectra
    else:
        raise ValueError("No valid data provided.")

    # Remove background
    if background is not None:
        a, b = background
        spectrum_image = spectrum_image.remove_background((a, b))

    # Crop the energy range
    if crop == None:
        c = spectrum_image.axes_manager[-1].axis[0]
        d = spectrum_image.axes_manager[-1].axis[-1]
        print('No crop applied.')
    else: 
        c, d = crop
        spectrum_image = spectrum_image.isig[c:d]

    # Convert the spectrum to a numpy array
    spectrum_data = spectrum_image.data

    # Normalize the data if requested
    if normalize:
        spectrum_data_ = spectrum_data.copy()
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_, _ = sklearn.preprocessing.normalize(spectrum_data_,norm = 'max',return_norm=True) #normalize
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
    else:
        spectrum_data_ = spectrum_data.copy()
        
    # Check if the dimensions of the input spectrum match the training data dimensions
    if spectrum_data.shape[-1] < ref_spectra.shape[-1]:
        print(f"The dimensions of the SI provided ({spectrum_data.shape[-1]}) is smaller than the training data ({ref_spectra.shape[-1]}).")
        print("It is recommended to reduce the training data resolution and not the evaluated Spectrum Image.")
    
    #----------------------------------------------------------------------#    
    #eje de energias de los espectros facilitados:
    Eaxis_si = spectrum_image.axes_manager[-1].axis
    #eje de energias de los datos de referencia: 
    Eaxis_ref = np.arange(c,d,(d-c)/ref_spectra.shape[-1])
    
    if Eaxis_si.shape[-1] != Eaxis_ref.shape[-1]:
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data.shape[-1])) #reshape
        spectrum_data_ = reduce_energy_resolution(spectrum_data_,eje_original=Eaxis_si,eje_nuevo=Eaxis_ref)
        spectrum_data_ = spectrum_data_.reshape((spectrum_data.shape[0],spectrum_data.shape[1],spectrum_data_.shape[-1])) #reshape
    else: 
        pass
    
    #----------------------------------------------------------------------#
    #Plot the spectra:
    plt.figure(dpi=150)
    plt.title('Comparative of training spectra and SI spectra')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(Eaxis_ref,ref_spectra[idx],linewidth=0.5,color='red')
    idx = np.random.randint(0,ref_spectra.shape[0])
    plt.plot(Eaxis_ref,ref_spectra[idx],linewidth=0.5,color='blue')
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(Eaxis_ref,spectrum_data_[idx][idy],linewidth=0.5,color='fuchsia',label='x: {}, y: {}'.format(idx,idy))
    idx = np.random.randint(0,spectrum_data_.shape[0]); idy = np.random.randint(0,spectrum_data.shape[1])
    plt.plot(Eaxis_ref,spectrum_data_[idx][idy],linewidth=0.5,color='purple',label='x: {}, y: {}'.format(idx,idy))
    plt.legend()
    plt.show()
    #-----------------------------------------------------------------------------------#
    
    #--------------------------Clustering for background--------------------------------#    
    labels, etiquetas = kmeans_clustering(spectrum_data_,n_cluster=2,norma=None)
    
    #Comprovar que el clustering funcione bien: 
    _, cuentas = np.unique(labels,return_counts=True)
    for i in cuentas: 
        if i < (labels.shape[0]*labels.shape[1])*0.05:
            plt.figure(dpi=100)
            plt.title('Clustering for background')
            plt.imshow(labels)
            plt.show()
            plt.figure(dpi=100)
            plt.title('Clustering for centroids')
            plt.plot(etiquetas[0])
            plt.plot(etiquetas[1])
            plt.show()    
            raise ValueError("The clustering has not been able to separate the background from the signal.")
        else: 
            pass
    
    #Vamos a comparar con el punto de fondo: 
    x_back, y_back = back_point
    #value es el indicador del label del background
    if labels[x_back][y_back] == 0:
        value = 1
    else:
        value = 0
    #-----------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------#
    spectrum_labeled = spectrum_data_.reshape(spectrum_data.shape[0]*spectrum_data.shape[1],spectrum_data_.shape[-1])
    spectrum_labeled_ = spectrum_data_[labels==value]
    #-----------------------------------------------------------------------------------#    
    
    #----------------------------Clasificacion------------------------------------------#
    # Clasificador probabilistico:
    if probability:
        clasificacion_0 = model.predict_proba(spectrum_labeled)
    # Clasificador binario: 
    clasificacion_1 = model.predict(spectrum_labeled_)
    num_classes = model.classes_.shape[0]

    #-----------------Ajustar clasificacion binaria al background-----------------------#
    labels_1 = labels.copy()
    count = 0 
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels_1[x][y] == value:
                labels_1[x][y] = clasificacion_1[count]
                count+=1
            else: 
                labels_1[x][y] = -1
    #-----------------------------------------------------------------------------------#
    
    #---------------------------------Color_map-----------------------------------------#
    cmap_original = plt.get_cmap("plasma_r")
    custom_cmap = cmap_original.copy()
    # Define el color negro para valores por debajo del limite inferior
    custom_cmap.set_under('black')
    custom_norm = mcolors.Normalize(vmin=-0.01, vmax=num_classes-1)
    custom_norm_ = mcolors.Normalize(vmin=-0.01, vmax=1.)
    #-----------------------------------------------------------------------------------#

    fig_ = []
    #-----------------------------------------------------------------------------------#
    if probability: 
        clasificacion_0 = clasificacion_0.reshape(spectrum_data.shape[0],spectrum_data.shape[1],clasificacion_0.shape[-1])
        clasificacion_0_ = clasificacion_0.copy()
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                if labels[x][y] == value:
                    pass
                else:
                    clasificacion_0_[x,y,:] = -1
        
        for i in range(num_classes):
        # Calcular el valor de las classes ponderado
            fig = plt.figure(dpi=200)
            fig_.append(fig)
            im = plt.imshow(clasificacion_0_[:,:,i], cmap=custom_cmap, norm=custom_norm_)
            cbar = plt.colorbar(im, label='Probability')
            plt.title('Class_{}'.format(i))
            cbar.set_ticks(np.arange(0,1.05,0.25))
            cbar.set_ticklabels(np.arange(0,1.05,0.25))
            plt.xticks([])
            plt.yticks([])
            scale_length = spectrum_image.axes_manager['x'].scale
            scale_unit = spectrum_image.axes_manager['x'].units
            #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
            #plt.gca().add_artist(sb)
        """
        #Code to plot the oxidation state: (have to be adapted to the oxide)
        classes_value = np.sum([(i + 2)*clasificacion_0[..., i] for i in range(num_classes)], axis=0)
        custom_norm_ox = mcolors.Normalize(vmin=2, vmax=3)
        # Calcular el valor de las classes ponderado
        fig = plt.figure(dpi=200)
        fig_.append(fig)
        im = plt.imshow(classes_value, cmap=custom_cmap, norm=custom_norm_ox)
        cbar = plt.colorbar(im, label='Oxidation State')
        #plt.title('Class_{}'.format(i))
        cbar.set_ticks(np.arange(2.,3.5,1.))
        cbar.set_ticklabels(np.arange(2,3.5,1))
        plt.xticks([])
        plt.yticks([])
        scale_length = spectrum_image.axes_manager['x'].scale
        scale_unit = spectrum_image.axes_manager['x'].units
        #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
        #plt.gca().add_artist(sb)
        """
    #-----------------------------------------------------------------------------------#
    
    #----------------------------Binary clasification-----------------------------------#
    #clasificacion_1 = clasificacion_1.reshape(spectrum_data.shape[0],spectrum_data.shape[1])
    clasificacion_1 = labels_1
    #-----------------------------------------------------------------------------------#
    
    #-----------------------------------------------------------------------------------#        
    fig1 = plt.figure(dpi=200)
    im = plt.imshow(clasificacion_1, cmap=custom_cmap, norm=custom_norm)
    cbar = plt.colorbar(im,label='Classes')
    plt.title('Binary classifier')
    cbar.set_ticks(np.arange(0,num_classes))
    cbar.set_ticklabels(np.arange(0,num_classes))
    plt.xticks([])
    plt.yticks([])
    scale_length = spectrum_image.axes_manager['x'].scale
    scale_unit = spectrum_image.axes_manager['x'].units
    #sb = ScaleBar(scale_length, scale_unit, box_alpha=0, color='yellow', location=0,length_fraction=0.4,height_fraction=0.5,width_fraction=0.015,scale_loc='top',font_properties={'size':14})
    #plt.gca().add_artist(sb)
    #-----------------------------------------------------------------------------------#   
    
    #-----------------------------------------------------------------------------------#
    num_classes = model.classes_.shape[0]
    mean_spectra = np.zeros((num_classes, spectrum_data.shape[-1]))

    for i in range(num_classes):
        indices = np.where(clasificacion_1 == i)
        class_spectra = spectrum_data[indices[0], indices[1]]
        mean_spectra[i] = np.mean(class_spectra, axis=0)
    #-----------------------------------------------------------------------------------#  
    fig2 = plt.figure(dpi=300)
    for i in range(0,num_classes):
        plt.plot(Eaxis_ref,mean_spectra[i],label='{}'.format(i),linewidth=0.5,alpha=0.8)
    #plt.xlim(690.,750.)
    plt.title('Average spectrum of each class')
    plt.legend(title='Class') 
    
    #-----------------------------------------------------------------------------------#
    if figure_name is not None:
        if probability:
            cont = 0
            for figure in fig_:
                figure.savefig(figure_name + '_probs{}'.format(cont),dpi=500)
                cont += 1
        fig1.savefig(figure_name + '_binary',dpi=500)
        fig2.savefig(figure_name + '_centroids',dpi=500)
    #----------------------------------------------------------------------------------------------------------------#
    return None