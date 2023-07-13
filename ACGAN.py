#ACGAN code:

#Tensorflow stuff:
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, Conv1DTranspose, ReLU, LeakyReLU, GaussianNoise, Add
from tensorflow.keras.utils import to_categorical as np_utils

#otros modulos:
import time
import scipy
import numpy as np
import pandas as pd
from IPython import display


from DanPy.gan_layers import MinibatchDiscrimination, calculate_fid, Translation, PoisonNoise



class ACGAN():
    """
    Clase para hacer un ACGAN donde en las perdidas tambien se tiene en cuenta el label. 
    """
    
    def __init__(self, g_model, d_model, \
                 dataset, categories, \
                 latent_dim, batch_size=256, num_classes=5, eval_size=25, buffer_size=16\
                ):

        #BASIC STUFF:
        self.generator = g_model
        self.discriminator = d_model        
        self.dataset = dataset
        self.categories = categories

        #Num clases:
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.n_batch = batch_size
        self.eval_size = eval_size
        self.buffer_size = buffer_size
        
        #Reduccion del learning rate para evitar overfitting.
        self.decay_rate = 0.1
        self.decay_steps = 1000
        
        #ADAM optimizadores: 
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-3)
        
        #Funciones de perdida: 
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def show_fakes(self, number = 10):
        """
        Method to plot the spectra. 
        """
        noise = tf.random.normal([number, latent_dim])
        label = tf.random.uniform([number, 1], minval=0, maxval=5, dtype = tf.int32)
        label = np_utils.to_categorical(label, num_classes = self.num_classes)
        noise = tf.random.normal([number, self.latent_dim])

        gen_img = self.generator.predict([noise,label])
        
        # Display fake (generated) images
        fig, axs = plt.subplots(3, 3, sharey=False,tight_layout=True, figsize=(12,6), facecolor='white')
        k = 0
        for i in range(0,3):
            for j in range(0,3):
                axs[i,j].plot(gen_img[k])
                axs[i,j].set(title=label[k])
                axs[i,j].axis('on')
                k=k+1
        plt.show()
        
    #compilar la GAN: 
    def compile_(self, d_opt):
        # Compile the generator and discriminator models
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])
    
    #-----------------------------------------------------------------------------------------#
    #                                  LOSSES STRATEGY: 
    #-----------------------------------------------------------------------------------------#

    def discriminator_loss(self, real_output, fake_output, y_true, y_pred):
        """
        Discriminator losses
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output)-0.1, real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        class_loss = self.cat_cross_entropy(y_true, y_pred)
        total_loss = real_loss + fake_loss + class_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        """
        Generator losses
        """
        gen_loss = self.cross_entropy(tf.ones_like(fake_output)-0.1, fake_output)
        return gen_loss
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    #EVALUATION STRATEGY:
    def calculate_fid(self, real_data, fake_data):
        """
        Function to calculate the Fréchet Inception Distance (FID). 
        """
        mean1, sigma1 = real_data.mean(axis=0), np.cov(real_data, rowvar=True)
        mean2, sigma2 = fake_data.mean(axis=0), np.cov(fake_data, rowvar=True)
        sum_sq_diff = np.sum((mean1 - mean2)**2)
        cov_mean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2.0*cov_mean)
        return fid
    #-----------------------------------------------------------------------------------------#

    #-----------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------#
    @tf.function
    def train_step(self,images):
        
        noise = tf.random.normal([self.n_batch, self.latent_dim])
        img, img_lab = images
        
        #img = tf.convert_to_tensor(img, dtype=tf.float64)
        #img_lab = tf.convert_to_tensor(img_lab, dtype=tf.int32)
        print()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator([noise, img_lab], training=True)
    
            real_output, y_pred = self.discriminator([img, img_lab], training=False)
            fake_output, _ = self.discriminator([generated_images, img_lab], training=False)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output, img_lab, y_pred)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        #gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Calcula la tasa de aprendizaje dinámica
        learning_rate_g = self.generator_optimizer.learning_rate*\
        (self.decay_rate**(tf.cast(self.generator_optimizer.iterations, tf.float32)/self.decay_steps))
        self.generator_optimizer.learning_rate.assign(learning_rate_g)
        # Calcula la tasa de aprendizaje dinámica
        #learning_rate_d = self.discriminator_optimizer.learning_rate*\
        #(self.decay_rate**(tf.cast(self.discriminator_optimizer.iterations, tf.float32)/self.decay_steps))
        #self.discriminator_optimizer.learning_rate.assign(learning_rate_d)
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        #self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss#, disc_loss
    #-----------------------------------------------------------------------------------------#    
    @tf.function
    def train_step_d(self,images):
    
        noise = tf.random.normal([self.n_batch, self.latent_dim])
        img, img_lab = images
    
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator([noise, img_lab], training=False)
    
            real_output, y_pred = self.discriminator([img, img_lab], training=True)
            fake_output, _ = self.discriminator([generated_images, img_lab], training=True)
    
            disc_loss = self.discriminator_loss(real_output, fake_output, img_lab, y_pred)
    
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Calcula la tasa de aprendizaje dinámica
        learning_rate_d = self.discriminator_optimizer.learning_rate*\
        (self.decay_rate**(tf.cast(self.discriminator_optimizer.iterations, tf.float32)/self.decay_steps))
        self.discriminator_optimizer.learning_rate.assign(learning_rate_d)
        
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss
    #-----------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------#


    def train_modified(self,n_epochs):
        
        g_model = self.generator
        d_model = self.discriminator
        
        self.compile_(self.discriminator_optimizer)
        
        losses = []
        fid_tol = 2.
        
        #Datos entrada:
        dataset = np.array(self.dataset)
        categories = np.array(self.categories)

        #Generar los labels: 
        sampled_labels_0 = tf.random.uniform(shape=(self.eval_size, 1), minval=0, maxval=1, dtype=tf.int32)
        sampled_labels_1 = tf.random.uniform(shape=(self.eval_size, 1), minval=1, maxval=2, dtype=tf.int32)
        sampled_labels_2 = tf.random.uniform(shape=(self.eval_size, 1), minval=2, maxval=3, dtype=tf.int32)
        sampled_labels_3 = tf.random.uniform(shape=(self.eval_size, 1), minval=3, maxval=4, dtype=tf.int32)
        sampled_labels_4 = tf.random.uniform(shape=(self.eval_size, 1), minval=4, maxval=5, dtype=tf.int32)
        
        # Convertir los labels a una distribución categórica:
        sampled_labels_0 = tf.one_hot(sampled_labels_0, depth=5)
        sampled_labels_1 = tf.one_hot(sampled_labels_1, depth=5)
        sampled_labels_2 = tf.one_hot(sampled_labels_2, depth=5)
        sampled_labels_3 = tf.one_hot(sampled_labels_3, depth=5)
        sampled_labels_4 = tf.one_hot(sampled_labels_4, depth=5)
        
        # Eliminar las dimensiones de tamaño 1:
        sampled_labels_0 = tf.squeeze(sampled_labels_0, axis=[1])
        sampled_labels_1 = tf.squeeze(sampled_labels_1, axis=[1])
        sampled_labels_2 = tf.squeeze(sampled_labels_2, axis=[1])
        sampled_labels_3 = tf.squeeze(sampled_labels_3, axis=[1])
        sampled_labels_4 = tf.squeeze(sampled_labels_4, axis=[1])
        
        #Separar datos por categoria: 
        x_0 = dataset[categories==0]
        x_1 = dataset[categories==1]
        x_2 = dataset[categories==2]
        x_3 = dataset[categories==3]
        x_4 = dataset[categories==4]
        
        #Valor minimo: 
        shape = np.array([x_0.shape[0],x_1.shape[0],x_2.shape[0],x_3.shape[0],x_4.shape[0]])
        
        #Categorical labels:
        categories = np_utils.to_categorical(categories, num_classes=self.num_classes)

        batch_count = 0
        for i in range(n_epochs): 
                        
            ##--------------------------------------##
            start = time.time()
            
            dataset_ = tf.data.Dataset.from_tensor_slices((dataset,categories)).shuffle(self.buffer_size).batch(self.n_batch)
            gen_loss_0, dis_loss_0, dis_loss_1 = [], [], []
    
            ##--------------------------------------##
            #Entrenar solo el discriminador:
            for image_batch in dataset_: 
                dis_loss = self.train_step_d(image_batch)
                dis_loss_1.append(dis_loss.numpy())
                batch_count += 1
            #Entrenar solo generador: 
            for image_batch in dataset_:
                gen_loss = self.train_step(image_batch)
                gen_loss_0.append(gen_loss.numpy())
                batch_count += 1 
            #Entrenar solo el discriminador:
            for image_batch in dataset_: 
                dis_loss = self.train_step_d(image_batch)
                dis_loss_0.append(dis_loss.numpy())
                batch_count += 1 
    
            gen_loss_0 = np.array(gen_loss_0).mean()
            dis_loss_0 = np.array(dis_loss_0).mean()
            dis_loss_1 = np.array(dis_loss_1).mean()
            
            # ---------------------
            #  Evaluate GAN:
            # ---------------------        
            # Seleccionamos datos para los indices:
            indices = np.random.randint(0,np.min(shape),self.eval_size)
            real_0 = x_0[indices]
            real_1 = x_1[indices]
            real_2 = x_2[indices]
            real_3 = x_3[indices]
            real_4 = x_4[indices]
            
            #Generamos fake data: 
            noise_eval =  tf.random.normal((self.eval_size,self.latent_dim),mean=0.,stddev=1.)
            
            gen_img_0 = g_model.predict([noise_eval,sampled_labels_0])
            gen_img_1 = g_model.predict([noise_eval,sampled_labels_1])
            gen_img_2 = g_model.predict([noise_eval,sampled_labels_2])
            gen_img_3 = g_model.predict([noise_eval,sampled_labels_3])
            gen_img_4 = g_model.predict([noise_eval,sampled_labels_4])
            
            aux_fid = []
            aux_fid.append(self.calculate_fid(real_0,gen_img_0))
            aux_fid.append(self.calculate_fid(real_1,gen_img_1))
            aux_fid.append(self.calculate_fid(real_2,gen_img_2))
            aux_fid.append(self.calculate_fid(real_3,gen_img_3))
            aux_fid.append(self.calculate_fid(real_4,gen_img_4))
            
            fid_value = np.mean(np.array(aux_fid))
            
            time_ =  time.time() - start
            print('Contador:',batch_count,'gen_losses:',gen_loss_0,'dis_losses:',dis_loss_0,'tiempo',time_,'s')
            print('Epoch: %d, fid value: %.5f'%(i+1,fid_value))
            
            if fid_value <= fid_tol:
                print('Epoch: %d, Ending fid value: %.5f'%(i+1,fid_value))
                return None
            
            # Summarize training progress and loss
            if batch_count % 5 == 0:
                display.clear_output(wait=True)
                total_time = time.time() - start
                print('Epoch: %d, Batch_Contador: %d, D_Loss=%.5f, Gen_Loss=%.5f'%(i+1, batch_count, dis_loss_0, gen_loss_0))
                print('Epoch: %d, Batch_Contador: %d, D_Loss_1=%.5f'%(i+1, batch_count, dis_loss_1))
                #print('Epoch: %d, Batch_Contador: %d,Time so far'%(i+1, batch_count,total_time))
                self.show_fakes(number=10)





