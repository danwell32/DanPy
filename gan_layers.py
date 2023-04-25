import tensorflow as tf
import keras
from keras import backend
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import scipy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


#CLASES DE LAS LAYERS COSTUMS: 

class PoisonNoise(keras.layers.Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    Args:
        std: Float, standard deviation of the noise distribution.
        seed: Integer, optional random seed to enable deterministic behavior.
    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as input.
    """
    def __init__(self, lam, std,**kwargs):
        super(PoisonNoise, self).__init__(**kwargs)
        #super(PoisonNoise, self).all(*args, **kwargs)
        #self.supports_masking = True
        self.lam = lam
        self.std = std
    def call(self, inputs, training=None):
        def noised():
            return inputs + (tf.random.poisson(shape=tf.shape(inputs),lam=self.lam,dtype=inputs.dtype)*self.std)
        return backend.in_train_phase(noised, inputs, training=training)
    """
    def get_config(self):
        config = {'lam': self.lam,'std:': self.std}
        base_config = super(PoisonNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    """
    def get_config(self):
        config = super(PoisonNoise, self).get_config()
        config.update({'lam': self.lam,'std':self.std})
        return config

class MinibatchDiscrimination(tf.keras.layers.Layer):

    def __init__(self, num_kernel, dim_kernel,kernel_initializer='glorot_uniform', **kwargs):
        self.num_kernel = num_kernel
        self.dim_kernel = dim_kernel
        self.kernel_initializer = kernel_initializer
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.num_kernel*self.dim_kernel),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(MinibatchDiscrimination, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        activation = tf.matmul(x, self.kernel)
        activation = tf.reshape(activation, shape=(-1, self.num_kernel, self.dim_kernel))
        #Mi
        tmp1 = tf.expand_dims(activation, 3)
        #Mj
        tmp2 = tf.transpose(activation, perm=[1, 2, 0])
        tmp2 = tf.expand_dims(tmp2, 0)
        
        diff = tmp1 - tmp2
        
        l1 = tf.reduce_sum(tf.math.abs(diff), axis=2)
        features = tf.reduce_sum(tf.math.exp(-l1), axis=2)
        return tf.concat([x, features], axis=1)
        
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.num_kernel)
      
    def get_config(self):
        config = super().get_config()
        config['dim_kernel'] =  self.dim_kernel
        config['num_kernel'] = self.num_kernel
        config["kernel_initializer"] = self.kernel_initializer
        return config

class Translation(keras.layers.Layer):
    """Apply a translation of lentgh trans, 
        solo la aplica al 50% de los datos.
        Las aplica hacia izquierda y derecha indistintamente. 
    Args:
        trans: Integer, shift.
    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode (doing nothing).
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as input.
    """
    def __init__(self, trans,**kwargs):
        super(Translation, self).__init__(**kwargs)
        self.trans = trans
    def call(self, inputs, training=None):
        def translated():
            #Dimensiones necesarias para la traslacion: 
            trans = int(self.trans)
            n_features = int(inputs.get_shape()[-1])
            val = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)            
            inputs_ = tf.cast(inputs,tf.float32)
            if val%2==0: #Traslacion hacia la izquierda. 
                inputs_initial = inputs_[:,trans:n_features]
                final = inputs_[:,n_features-trans:n_features]
                final = tf.cast(final,tf.float32)
                return tf.concat((inputs_initial,final),axis=1)
            else: #Traslacion hacia la derecha.
                inputs_final = inputs_[:,0:n_features-trans]
                initial = inputs_[:,0:trans]
                initial = tf.cast(initial,tf.float32)
                return tf.concat((initial,inputs_final),axis=1)
        return backend.in_train_phase(translated, inputs, training=training)    
    def get_config(self):
        config = super().get_config()
        config['trans'] =  self.trans
        return config
    #OLD GET_CONFIG
    #def get_config(self):
    #    config = {'trans': self.trans}
    #    base_config = super(Translation, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))

"""
class Translation_(keras.layers.Layer):
    Apply a translation of lentgh trans, 
        solo la aplica al 50% de los datos.
        Las aplica hacia izquierda y derecha indistintamente. 
    Args:
        trans: Integer, shift.
    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode (doing nothing).
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as input.
    
    def __init__(self, trans,**kwargs):
        super(Translation, self).__init__(**kwargs)
        self.trans = trans
    def call(self, inputs, training=None):
        def translated():
            #Dimensiones necesarias para la traslacion: 
            trans = int(self.trans)
            n_features = int(inputs.get_shape()[-1])
            length = inputs.get_shape()[0]
            length_half = tf.cast(tf.round(tf.divide(length,2)),tf.int32)
            val = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)            
            inputs_ = tf.cast(inputs,tf.float32)
            if val%2==0: #Traslacion hacia la izquierda. 
                inputs_initial = inputs_[:length_half,trans:n_features]
                final = inputs_[:length_half,n_features-trans:n_features]
                final = tf.cast(final,tf.float32)
                aux = tf.concat((inputs_initial,final),axis=1)
                return tf.concat((inputs_[length_half:,:],aux),axis=0)
            else: #Traslacion hacia la derecha.
                inputs_final = inputs_[length_half:,0:n_features-trans]
                initial = inputs_[length_half:,0:trans]
                initial = tf.cast(initial,tf.float32)
                aux = tf.concat((initial,inputs_final),axis=1)
                return tf.concat((inputs_[:length_half,:],aux),axis=0)
        return backend.in_train_phase(translated, inputs, training=training)
    def get_config(self):
        config = {'trans': self.trans}
        base_config = super(Translation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""

"""
#Prueba del funcionamiento de la traslacion. 
inputs = x_mn2[0:496,:]
inputs = tf.convert_to_tensor(inputs)
trans = 3
n_features = int(inputs.get_shape()[-1])
length = inputs.get_shape()[0]
length_half = tf.round(tf.divide(length,2))
length_half = tf.cast(length_half,tf.int32)
val = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
inputs = tf.cast(inputs,tf.float32)

if val%2==0: #Traslacion hacia la izquierda. 
    inputs_initial = inputs[:length_half,trans:n_features]
    final = inputs[:length_half,n_features-trans:n_features]
    final = tf.cast(final,tf.float32)
    a = tf.concat((inputs_initial,final),axis=1)
    a = tf.concat((inputs[length_half:,:],b),axis=0)
else: #Traslacion hacia la derecha.
    inputs_final = inputs[length_half:,0:n_features-trans]
    initial = inputs[length_half:,0:trans]
    initial = tf.cast(initial,tf.float32)
    b = tf.concat((initial,inputs_final),axis=1)
    b = tf.concat((inputs[:length_half,:],b),axis=0)
"""

def calculate_fid_tf(x, y):
    
    #Function to calculate the Fréchet Inception Distance (FID). 
    
    #x = tf.convert_to_tensor(x)
    #y = tf.convert_to_tensor(y)

    mean1 = tf.math.reduce_mean(x)
    sigma1 = tfp.stats.covariance(x)

    mean2 = tf.math.reduce_mean(y)
    sigma2 = tfp.stats.covariance(y)

    sum_sq_diff = tf.math.squared_difference(mean1,mean2)
    dot_product = tf.tensordot(sigma1,sigma2,axes=1)

    #Pasamos a complejos el producto, de esta manera evitamos que el sqrtm de error. 
    dot_product = tf.cast(dot_product, dtype=tf.complex64)
    cov_mean = tf.linalg.sqrtm(dot_product)
    cov_mean = tf.math.real(cov_mean)

    added = tf.math.add(sigma1,sigma2)
    added = tf.cast(added, dtype=tf.float32)

    multiply_by_2 = tf.math.multiply(cov_mean,2)
    multiply_by_2 = tf.cast(multiply_by_2, dtype=tf.float32)
    substraction = tf.math.subtract(added,multiply_by_2)
    trace = tf.linalg.trace(substraction)
    fid = tf.math.add(sum_sq_diff,trace)
    
    return fid

def calculate_fid(real_data, fake_data):
    """
    Function to calculate the Fréchet Inception Distance (FID).
    real_data: numpy array. 
    fake_data: numpy array.
    """
    mean1, sigma1 = real_data.mean(axis=0), np.cov(real_data, rowvar=True)
    mean2, sigma2 = fake_data.mean(axis=0), np.cov(fake_data, rowvar=True)
    sum_sq_diff = np.sum((mean1 - mean2)**2)
    cov_mean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2.0*cov_mean)
    return fid

def signaltonoise(data, axis=0, ddof=0):
    """
    Function to calculate the signal to noise ratio.
    data: numpy array. 
    """
    a = np.asanyarray(data)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd).mean()

def calculate_correlation(a,b):
    """
    Calculate the correlation between 2 arrays
    """
    correlation_mat = np.corrcoef(a, b)
    correlation_mat = np.triu(correlation_mat)
    correlation_mat = correlation_mat[(correlation_mat < 0.999) & (correlation_mat > 0.0)]
    return np.mean(correlation_mat)

def calculate_cos_dist(a,b):
    """
    Calculate the Cosine distance 2 arrays
    """
    cos_matrix = cosine_distances(a,b)
    cos_matrix = cos_matrix[cos_matrix > 0.0].flatten()
    return np.mean(cos_matrix)

def translation(inputs,translation):
    """
    Apply traslations in tensors 
    """
    #Dimensiones necesarias para la traslacion:
    inputs = tf.convert_to_tensor(inputs)
    trans = int(translation)
    n_features = int(inputs.get_shape()[-1])
    val = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
    if val%2==0: #Traslacion hacia la izquierda. 
        inputs_initial = inputs[:,trans:n_features]
        inputs_initial = tf.cast(inputs_initial,tf.float64)
        final = inputs[:,n_features-trans:n_features]
        final = tf.cast(final,tf.float64)
        return tf.concat((inputs_initial,final),axis=1)
    else: #Traslacion hacia la derecha.
        inputs_final = inputs[:,0:n_features-trans]
        inputs_final = tf.cast(inputs_final,tf.float64)
        initial = inputs[:,0:trans]
        initial = tf.cast(initial,tf.float64)
        return tf.concat((initial,inputs_final),axis=1)
    
def poisson_noise(inputs,lam,std):
    """
    Apply noise in tensors 
    """
    return inputs + tf.random.poisson(shape=tf.shape(inputs),lam=lam,dtype=inputs.dtype)*std

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(dpi=200)
    for i in range(predictions.shape[0]):
        plt.subplot()
        plt.plot(predictions[i],linewidth=.1,alpha=.8)
        plt.axis('on')
    #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def graficar_resultados(spectra,latent_dim,generator,scatter=False):
    """
    Function to evaluate viasually the generated spectra.
    """
    data = spectra
    dpi = 200
    constante=1
    n_spectra=100
    a = 600
    b = 800
    ##----------------------------------##
    #Generate data: 
    generated_data = generator.predict(tf.random.normal([n_spectra, latent_dim]))
    generated_data = generated_data*constante
    #Define the energy axis: 
    #generated_data = sklearn.preprocessing.normalize(generated_data,'max')
    
    x_axis = np.linspace(a, b, generated_data.shape[1])
    
    #Large quantity of spectra by image: 
    fig1 = plt.figure(dpi=dpi)
    for i in range(0,generated_data.shape[0]):
        if scatter:
            plt.scatter(x_axis,generated_data[i],s=1.)
        else:
            plt.plot(x_axis,generated_data[i],linewidth=.3,linestyle=':')
            
    #plt.ylim((-.1,1.0))
    plt.xlim((a,b))
    
    ##----------------------------------##
    red_patch = mpatches.Patch(color='red', label='Generator Losses')
    purple_patch = mpatches.Patch(color='purple', label='Discriminator Losses')
    
    #Comparate spectra from original: 
    real_spec = data.copy()
    set_ind = np.random.randint(0, data.shape[0], 5)
    originals = real_spec[set_ind]    
    gen_spec = generator.predict(np.random.normal(0, 1,(5, latent_dim)))
    gen_spec = gen_spec*constante
    #gen_spec = sklearn.preprocessing.normalize(gen_spec,'max')
    all_spec = np.concatenate((originals,gen_spec),axis=0)
    
    fig4 = plt.figure(dpi=dpi)
    i=0
    for i in range(1,all_spec.shape[0]):
        if scatter:
            plt.scatter(x_axis,all_spec[i]+i,s=1.,color='red')
        else:
            plt.plot(x_axis,all_spec[i]+i,linewidth=.5,linestyle=':',color='red')
        plt.hlines(y=i,xmin=a,xmax=b,linewidth=.2,linestyle=':',color='black')
        
    plt.hlines(y=5.,xmin=a,xmax=b,linewidth=.5,linestyle=':',color='black')
    
    # Spectrum lines: 
    n_batch = 150
    real_spec = data.copy()
    set_ind = np.random.randint(0, data.shape[0], n_batch)
    originals = real_spec[set_ind]    
    gen_spec = generator.predict(np.random.normal(0, 1,(n_batch, latent_dim)))
    #gen_spec = sklearn.preprocessing.normalize(gen_spec,'max')
    all_spec = np.concatenate((originals,gen_spec),axis=0)
    gen_spec = gen_spec*constante
    
    fig5 = plt.figure(figsize=(5,50),dpi=dpi)
    plt.imshow(all_spec,cmap='viridis')
    ax = plt.gca()
    plt.xlabel('Energy loss eV')
    plt.yticks([])    
    plt.show()
    return None














def gan_train(discriminator, 
              generator, 
              gan, 
              spectra, 
              spectra_info = {'init_energy':460.,'end_energy':800.},
              training_parameters = {'latent_dim':30,'n_epochs':5000,'n_batches':32,'mode':2},
              stopping_criterias = {'FID_tol':2.,'correlation_tol':.7,'cosine_tol':0.2},
              verbose=5):
    """
    Function to train by batches a Generative Adversarial Network.
    discriminator:
    generator :
    gan: 
    spectra: numpy array
    spectra_info: dict
    training_parameters: dict
    stopping_criterias: dict
    verbose:
    ---------------
    Return: 
    Trained discriminator: trained tensorflow model.
    Trained generator: trained tensorflow model.
    loss_values: list 
    """
    #Variables para visualizar:
    show = True #mostrar la estructura de la red
    
    #Initialize training parameters:
    latent_dim = training_parameters['latent_dim'];
    n_batch = training_parameters['n_batches'];
    n_epochs = training_parameters['n_epochs'];
    mode = training_parameters['mode'];
    #Stopping Criterias:
    fid_tol = stopping_criterias['FID_tol'];
    correlation_tol = stopping_criterias['correlation_tol'];
    cos_tol = stopping_criterias['cosine_tol'];

    #List to generate a gif from the training and to save the loss values: 
    gif = []; loss_values = [];

    #Initialize the epochs:
    epoch = 0;
    
    spectra = tf.convert_to_tensor(spectra,dtype=tf.float32);
    
    #Check that the dimensions of dataset given is right:
    if len(spectra.get_shape()) > 2: 
        raise Exception('The training dataset has to be 2 dimensional')
    
    evaluation = int(spectra.get_shape()[-1]/5); #Take a fifth part of the whole dataset to calculate fid distance, correlation and cosine distance.

    #Spectrum dimension:
    input_dim = spectra.get_shape()[-1]
    
    #Generate the energy axis: 
    e_0 = spectra_info['init_energy']
    e_f = spectra_info['end_energy']
    x_axis = np.linspace(e_0, e_f, spectra.get_shape()[-1]) #NUMPY
    
    # Define the discriminator model:
    discriminator = discriminator(input_dim)
    if show:
        print(discriminator.summary())
    
    # Define the generator model:
    generator = generator(latent_dim,input_dim)
    if show: 
        print(generator.summary())
        
    # Define the GAN model:
    gan = gan(generator, discriminator)
    #    if show: 
    #        print(generator.summary())
    
    
    #Take a sixth part of the whole dataset to calculate fid distance and correlation
    shape = tf.random.uniform((n_batch,))
    shape_eval = tf.random.uniform((evaluation,))
    
    ##--------------------------------------##
    # Soft labels to evaluate fake & real data:
    y_real_eval = tf.ones_like((shape_eval))-0.1
    y_fake_eval = tf.zeros_like((shape_eval))+0.1
    # Strick Labels for fake & real data:
    #y_real_eval = tf.ones_like((shape_eval)) # Real data = 1
    #y_fake_eval = tf.zeros_like((shape_eval)) # Fake data = 0
    ##--------------------------------------##
    
    ##--------------------------------------##
    # Soft labels for fake & real data:
    y_real = tf.ones_like((shape))-0.1 # Real data = 0.9
    y_fake = tf.zeros_like((shape))+0.1 # Fake data = 0.1
    # Strick Labels for fake & real data:    
    #y_real = tf.ones_like((shape)) # Real data = 1
    #y_fake = tf.zeros_like((shape)) # Fake data = 0    
    ##--------------------------------------##

    start = time.time() #Time counter
    
    #Bucle:
    while epoch < n_epochs:
        start_time = time.time() #Time counter for epoch
        
        ##-----Discriminator first training-----##
        idx = tf.random.uniform((n_batch,),0,spectra.get_shape()[0],dtype=tf.int32)
        real_imgs = tf.gather(spectra,idx.numpy())
        noise = tf.random.normal((n_batch, latent_dim),mean=0.,stddev=1.) #Generate a normal distribution to feed the generator.
        generated_images = generator.predict(noise) #Generate fake data.
        #Train the discriminator: 
        generator.trainable = False; discriminator.trainable = True;
        d_loss_r = discriminator.train_on_batch(real_imgs,y_real) #Real data losses
        d_loss_f = discriminator.train_on_batch(generated_images,y_fake) #Fake data losses
        d_loss = (d_loss_r[0] + d_loss_f[0]) #Discriminador total loss 
        ##---------------------------------------#
        
        ##-----Generator training-----##
        generator.trainable = True; discriminator.trainable = False; 
        noise = tf.random.normal((n_batch, latent_dim),mean=0.,stddev=1.) #Generate a normal distribution to feed the generator.
        g_loss = gan.train_on_batch(noise,y_real) #Engañamos al generador le decimos que son datos reales.
        ##----------------------------##
        
        ##----Discriminator training with transformation(traslation)----##
        generator.trainable = False; discriminator.trainable = True;
        idy = tf.random.uniform((n_batch,),0,spectra.get_shape()[0],dtype=tf.int32)
        real_imgs_y = tf.gather(spectra,idy.numpy())
        noise = tf.random.normal((n_batch, latent_dim),mean=0.,stddev=1.)
        generated_imgs_ = generator.predict(noise)
        
        trans = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32) #traslacion que queramos hacer
        generated_img_trans = translation(generated_imgs_,trans.numpy())
        real_img_trans = translation(real_imgs_y,trans.numpy())
        
        d_loss_trans = discriminator.train_on_batch(real_img_trans,y_real) #Real data losses
        d_loss_trans = discriminator.train_on_batch(generated_img_trans,y_fake) #Fake data losses 
        d_loss_trans = (d_loss_trans[0] + d_loss_trans[0]) #Discriminador total loss
        ##-------------------------------------------------------------##
        
        #if mode == 3: #doble training discriminator
        #    ##-----Discriminator second training-----##
        #    generator.trainable = False; discriminator.trainable = True;
        #    idz = tf.random.uniform((n_batch,),0,spectra.get_shape()[0],dtype=tf.int32)
        #    real_imgs_z = tf.gather(spectra,idz.numpy())
        #    noise = tf.random.normal((n_batch, latent_dim),mean=0.0,stddev=1.0)
        #    generated_imgs_z = generator.predict(noise)
        #    
        #    d_loss_r_2 = discriminator.train_on_batch(real_imgs_z,y_real) #Real data losses
        #    d_loss_f_2 = discriminator.train_on_batch(generated_imgs_z,y_fake) #Fake data losses 
        #    d_loss_2 = (d_loss_r_2[0] + d_loss_f_2[0]) #Discriminador total loss
            ##---------------------------------------##
        
        ##-----Evaluation-----##
        idx_eval = tf.random.uniform((evaluation,),0,spectra.get_shape()[0],dtype=tf.int32)
        real_imgs_eval = tf.gather(spectra,idx_eval.numpy())
        noise_eval =  tf.random.normal((evaluation, latent_dim),mean=0.0,stddev=1.0)
        generated_imgs_eval = generator.predict(noise_eval)
        
        eva_real = discriminator.evaluate(real_imgs_eval,y_real_eval,verbose=0) #Real data accuracy
        eva_fake = discriminator.evaluate(generated_imgs_eval,y_fake_eval,verbose=0) #Fake data accuracy
        
        #FID distance: 
        fid_value = calculate_fid(real_imgs_eval.numpy(),generated_imgs_eval)
        #Correlation coeficient: 
        correlation_value = calculate_correlation(real_imgs_eval.numpy(),generated_imgs_eval)
        #Cosine distance:
        cosine_distance = calculate_cos_dist(real_imgs_eval.numpy(),generated_imgs_eval)
        
        ##--------------------##

        #Save generated data each 5 epochs:
        if epoch % 5 == 0:
            if mode == 3:
                d_loss_all = (d_loss + d_loss_2)/2
            else: 
                d_loss_all = d_loss
            loss_values.append([epoch,g_loss,d_loss_all,eva_fake,eva_real])
            gif.append([epoch,fid_value,correlation_value,cosine_distance])
            #gif.append(generated_images[np.random.randint(0,n_batch)])

        ##-----Stopping Criteria-----##
        """
        La idea es minimizar el error del generador. 
        Al generador lo engañamos para que piense que el noise que le damos son valores
        reales de los datos, pero es mentira, le estamos dando ruido. Entonces vamos a 
        iterar hasta que minimizamos este valor. 
        Por otra parte queremos maximizar el discriminador. 
        """
        if fid_value <= fid_tol and correlation_value >= correlation_tol and cosine_distance <= cos_tol:
            print('Final en epoch numero:',epoch,' FID:',fid_value,' Correlation:',correlation_value,' Cosine distance:',cosine_distance)
            #Plot las cosas:
            red_patch = mpatches.Patch(color='red', label='correlation')
            purple_patch = mpatches.Patch(color='green', label='cosine')
            gif = np.array(gif)
            plt.figure(dpi=200)
            plt.scatter(gif[:,0],gif[:,1],color='red')
            plt.xlabel('Epoch')
            plt.ylabel('FID')
            plt.figure(dpi=200)
            plt.scatter(gif[:,0],gif[:,2],color='red')
            plt.scatter(gif[:,0],gif[:,3],color='green')
            plt.legend(handles=[red_patch,purple_patch])    
            plt.xlabel('Epoch')
            plt.ylabel('Correlation&Cosine')
            return (discriminator,generator,loss_values,gif) #return a tuple
        ##---------------------------##
        
        divisor = int(n_epochs/(verbose*5))
        # Plot the progress:
        if epoch % divisor == 0:
            red_patch = mpatches.Patch(color='red', label='Generated data')
            blue_patch = mpatches.Patch(color='blue', label='Original data')
            #Plot single data point:
            k = np.random.randint(0,generated_imgs_eval.shape[0])
            plt.figure(dpi=200)  
            plt.scatter(x_axis, generated_imgs_eval[k],color='red', alpha=1.,s=1)
            plt.scatter(x_axis, spectra[k],color='blue', alpha=0.75, s=1)
            plt.legend(handles=[red_patch,blue_patch])
            plt.show()
            print('Number of epochs:',epoch,'\n')
            print('Frechet Inception Distance:',fid_value,'\n')
            print("Correlation: ", correlation_value,'\n')
            print("Cosine Distance", cosine_distance,'\n')

            print('Acc real:',eva_real,' Acc fake:',eva_fake)
            print('Discriminator losses:',d_loss_all,' Generator losses:',g_loss)
            
            time_taken = time.time() - start
            print('Time since start: %.2f min' % ((time.time() - start) / 60.0))
            print('Trained from step %i to %i in %.2f steps / sec' % (epoch-1, epoch, epoch/time_taken))
        epoch+=1
    
    red_patch = mpatches.Patch(color='red', label='correlation')
    purple_patch = mpatches.Patch(color='green', label='cosine')
    gif = np.array(gif)
    plt.figure(dpi=200)
    plt.scatter(gif[:,0],gif[:,1],color='red',s=2)
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.figure(dpi=200)
    plt.scatter(gif[:,0],gif[:,2],color='red',s=2)
    plt.scatter(gif[:,0],gif[:,3],color='green',s=2)
    plt.legend(handles=[red_patch,purple_patch])    
    plt.xlabel('Epoch')
    plt.ylabel('Correlation&Cosine')
    
    print('After ',epoch,' epochs the training function has been not able to find any proper GAN model')
    return None