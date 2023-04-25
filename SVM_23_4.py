import numpy as np
import hyperspy.api as hs
import sklearn.preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar

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

def translation(inputs, translation):
    """
    Apply translations on NumPy arrays.
    
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

def create_displaced_spectra(X, Y, num_displacements=100, samples_per_displacement=10, max_displacement=10):
    """
    Create displaced spectra by applying translations on the input arrays.
    
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

def classify_spectrum(spectra, model, ref_spectra, probability=True, normalize=True, background=None, crop=None, figure_name=None):
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
        spectrum_data_ = reduc_energy_resol(spectrum_data_,eje_original=Eaxis_si,eje_nuevo=Eaxis_ref)
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