
import time
from IPython import display
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_distances
import tensorflow as tf
from keras.layers import Flatten, Dense, GaussianNoise, Input, Concatenate, LeakyReLU, Dropout
from gan_layers import PoisonNoise, Translation, MinibatchDiscrimination

#ajkshdglsdhabhksdsda

# En esta version el generador puede segmentar el espectro en trozos iguales. 
class DGAN_3():

    def __init__(self,spectra, latent_dim, batch_size):
        
        self.spectra = spectra
        self.input_dim = spectra.shape[-1]
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_examples_to_generate = 6
        self.buffer_size = 12
        self.networks = False #summary gen-dis 
        
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001) #10-4
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005) #10-4
        
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        
        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    def discriminator_loss(self,real_output,fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output)*0.9, real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def translation(self,inputs,trans_val):
        """
        Apply traslations in tensors 
        """
        #Dimensiones necesarias para la traslacion:
        inputs = tf.convert_to_tensor(inputs)
        trans = int(trans_val)
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
    
    def calculate_fid(self,real_data, fake_data):
        """
        Function to calculate the Frechet Inception Distance (FID). 
        """
        mean1, sigma1 = real_data.mean(axis=0), np.cov(real_data, rowvar=True)
        mean2, sigma2 = fake_data.mean(axis=0), np.cov(fake_data, rowvar=True)
        sum_sq_diff = np.sum((mean1 - mean2)**2)
        cov_mean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2.0*cov_mean)
        return fid
    
    def calculate_correlation(self,a,b):
        """
        Calculate the correlation between 2 arrays.
        """
        correlation_mat = np.corrcoef(a, b)
        correlation_mat = np.triu(correlation_mat)
        #avoid diagonal values which are the correlation between tha same spectra
        correlation_mat = correlation_mat[(correlation_mat < 0.999) & (correlation_mat > 0.0)] 
        return np.mean(correlation_mat)
    
    def calculate_cos_dist(self,a,b):
        """
        Calculate the Cosine distance of 2 arrays.
        """
        cos_matrix = cosine_distances(a,b)
        cos_matrix = cos_matrix[cos_matrix > 0.0].flatten()
        return np.mean(cos_matrix)
    
    def generate_and_save_images(self, model, epoch, test_input):
        """
        Function to generate and save images.
        """
        save = False
        predictions = model(test_input, training=False)

        fig = plt.figure(dpi=200)
        for i in range(predictions.shape[0]):
            plt.subplot()
            plt.plot(predictions[i],linewidth=.1,alpha=.8)
            plt.axis('on')
        if save:
            fig.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def make_discriminator_model(self):
        
        n = 5
        lk = 0.3
        step = round(self.input_dim/n)
        
        entrada = Input(shape=(self.input_dim))
        x = Dense(self.input_dim, input_dim = self.input_dim)(entrada)
        x = LeakyReLU(lk)(x)
        x = Dropout(0.5)(x)
        
        for i in range(1,n):
            if i == 3:
                x = MinibatchDiscrimination(num_kernel=100, dim_kernel=5)(x)
            else:
                x = Dense((n-i)*step)(x)
                x = LeakyReLU(lk)(x)
                x = Dropout(0.5)(x)
        
        y = Dense(self.input_dim, input_dim = self.input_dim)(entrada)
        y = LeakyReLU(lk)(y)
        y = Dropout(0.5)(y)
        
        for i in range(1,n):
            if i == 3:
                y = MinibatchDiscrimination(num_kernel=150, dim_kernel=8)(y)
            else:
                y = Dense((n-i)*step)(y)
                y = LeakyReLU(lk)(y)
                y = Dropout(0.5)(y)
            
        z = Concatenate(axis=1)([x,y])
        z = Flatten()(z)
        z = Dense(1, activation='sigmoid')(z)
        model = tf.keras.Model(entrada,z)
        if self.networks:
            model.summary()
        return model

    def make_generator_model(self):

        num_segments = 6
        segment_outputs = []
        reduction_percentages = [0.15, 0.25, 0.5]
        trans = tf.random.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)
        step = round(self.input_dim/num_segments)
        std = 0.15
        lk = 0.3
        mult_scalar = 1.

        entrada = Input(shape=(self.latent_dim))
        pois = .5
        for i in range(num_segments):
            x = Dense(self.latent_dim)(entrada)
            x = LeakyReLU(lk)(x)
            x = PoisonNoise(lam=pois,std=1)(x)
            x = GaussianNoise(stddev=std)(x)
    
            for reduction_percentage in reduction_percentages:
                reduced_step = round(step * (1 - reduction_percentage))
                x = Dense(reduced_step)(x)
                x = LeakyReLU(lk)(x)
                x = PoisonNoise(lam=pois,std=1)(x)
                x = GaussianNoise(stddev=std)(x)
    
            x = Dense(step)(x)
            x = LeakyReLU(lk)(x)
            x = Translation(trans)(x)
    
            segment_outputs.append(x)
    
        pois = 0.15
        x_i = Concatenate()(segment_outputs)
        x_i = Dense(self.input_dim)(x_i)
        x_i = LeakyReLU(lk)(x_i)
        x_i = PoisonNoise(lam=pois, std=1)(x_i)
        x_i = Translation(trans)(x_i)
    
        x_f = Dense(self.input_dim, activation='tanh')(x_i)
        x_f = x_f * mult_scalar
    
        model = tf.keras.Model(entrada, x_f)
        if self.networks:
            model.summary()
        return model
    
    @tf.function #tf.function causes the function to be "compiled"
    def train_step_1(self,images):
        #Regular training of generator and discriminator. 
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
    
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss
        
    @tf.function #tf.function causes the function to be "compiled"
    def train_step_2(self,images):
        #Training of gan with translations.
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            #Apply the traslations.
            trans = tf.random.uniform(shape=(), minval=1, maxval=11, dtype=tf.int32) #traslacion que queramos hacer
            generated_images_trans = self.translation(generated_images,trans)
            images_trans = self.translation(images,trans)
            
            real_output = self.discriminator(images_trans, training=True)
            fake_output = self.discriminator(generated_images_trans, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss
    
    @tf.function #tf.function causes the function to be "compiled"
    def train_step_3(self,images):
        #Training of discriminator only. 
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=False)
            
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
    
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, epochs, stopping_criterias):

        gif = []; losses = []
        data = self.spectra
        data = tf.convert_to_tensor(data)
        seed = tf.random.normal([self.num_examples_to_generate, self.latent_dim])
        evaluation = self.batch_size

        #Stopping Criterias:
        fid_tol = stopping_criterias['FID_tol'];
        correlation_tol = stopping_criterias['correlation_tol'];
        cos_tol = stopping_criterias['cosine_tol'];
        
        for epoch in range(0,epochs):
            start = time.time()
            dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(self.buffer_size).batch(self.batch_size)
            
            for image_batch in dataset: #estandar sin mas
                gen_loss_1,dis_loss_1 = self.train_step_1(image_batch)
                
            for image_batch in dataset: #translaciones
                gen_loss_2,dis_loss_2 = self.train_step_2(image_batch) 
                
            for image_batch in dataset: #entrenamos el discriminador_0 2 veces
                gen_loss_3,dis_loss_3 = self.train_step_3(image_batch)
            
            print(gen_loss_1,dis_loss_1,'\n')
            print(gen_loss_2,dis_loss_2,'\n')
            print(gen_loss_3,dis_loss_3,'\n')
            
            losses.append([epoch,gen_loss_1.numpy(),dis_loss_1.numpy(),gen_loss_2.numpy(),dis_loss_2.numpy(),gen_loss_3.numpy(),dis_loss_3.numpy()])
            
            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator,epoch+1,seed)
            
            idx = tf.random.uniform((evaluation,),0,data.get_shape()[0],dtype=tf.int32)
            real_imgs = tf.gather(data,idx.numpy())
            noise_eval =  tf.random.normal((evaluation, self.latent_dim),mean=0.,stddev=1.)
            generated_imgs = self.generator.predict(noise_eval)

            #FID distance: 
            fid_value = self.calculate_fid(real_imgs.numpy(),generated_imgs)
            #Correlation coeficient: 
            correlation_value = self.calculate_correlation(real_imgs.numpy(),generated_imgs)
            #Cosine distance:
            cosine_distance = self.calculate_cos_dist(real_imgs.numpy(),generated_imgs)

            gif.append([epoch,fid_value,correlation_value,cosine_distance])
            print('Number of epochs:',epoch,'\n')
            print('Frechet Inception Distance:',fid_value,'\n')
            print("Correlation: ", correlation_value,'\n')
            print("Cosine Distance", cosine_distance,'\n')

            if fid_value <= fid_tol and correlation_value >= correlation_tol and cosine_distance <= cos_tol:# and epoch>200:
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
                return (self.discriminator,self.generator,gif,losses)
                        
        print('After ',epoch,' epochs the training function has been not able to find any proper GAN model')
        return None
    


###--------------------------------------------------------------------------------------##
###--------------------------------------------------------------------------------------##
class DGAN_2():
    
    def __init__(self,spectra, latent_dim, BATCH_SIZE):
        
        self.spectra = spectra
        self.input_dim = spectra.shape[-1]
        self.latent_dim = latent_dim
        self.BATCH_SIZE = BATCH_SIZE
        self.num_examples_to_generate = 6
        self.BUFFER_SIZE = 12
        
        self.generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.0001) #10-4
        self.discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.0005) #10-4
        
        self.generator = self.make_generator_model()
        self.discriminator_0 = self.make_discriminator_model_0()
        self.discriminator_1 = self.make_discriminator_model_1()

        # This method returns a helper function to compute cross entropy loss
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def discriminator_loss(self,real_output,fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output)-0.1, real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def translation(self,inputs,trans_val):
        """
        Apply traslations in tensors 
        """
        #Dimensiones necesarias para la traslacion:
        inputs = tf.convert_to_tensor(inputs)
        trans = int(trans_val)
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
        
    def calculate_fid(self,real_data, fake_data):
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
    
    def calculate_correlation(self,a,b):
        """
        Calculate the correlation between 2 arrays
        """
        correlation_mat = np.corrcoef(a, b)
        correlation_mat = np.triu(correlation_mat)
        correlation_mat = correlation_mat[(correlation_mat < 0.999) & (correlation_mat > 0.0)]
        return np.mean(correlation_mat)
    
    def calculate_cos_dist(self,a,b):
        """
        Calculate the Cosine distance 2 arrays
        """
        cos_matrix = cosine_distances(a,b)
        cos_matrix = cos_matrix[cos_matrix > 0.0].flatten()
        return np.mean(cos_matrix)
        
    def generate_and_save_images(self, model, epoch, test_input):
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
        
    def poisson_noise(self,inputs,lam,std):
        """
        Apply noise in tensors 
        """
        return inputs + tf.random.poisson(shape=tf.shape(inputs),lam=lam,dtype=inputs.dtype)*std
    
    #Este va genial:   
    def make_generator_model(self):
        num_segments = 6
        reduction_percentages = [0.15, 0.25, 0.5]
        trans = tf.random.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)
        step = round(self.input_dim / 4)
        std = 0.15
        lk = 0.3
        mult_scalar = 1.
        scale = True
        entrada = Input(shape=(self.latent_dim))
    
        pois = .5
        segment_outputs = []
    
        for i in range(num_segments):
            x = Dense(self.latent_dim)(entrada)
            x = LeakyReLU(lk)(x)
            x = PoisonNoise(lam=pois,std=1)(x)
            x = GaussianNoise(stddev=std)(x)
    
            for reduction_percentage in reduction_percentages:
                reduced_step = round(step * (1 - reduction_percentage))
                x = Dense(reduced_step)(x)
                x = LeakyReLU(lk)(x)
                x = PoisonNoise(lam=pois,std=1)(x)
                x = GaussianNoise(stddev=std)(x)
    
            x = Dense(step)(x)
            x = LeakyReLU(lk)(x)
            x = Translation(trans)(x)
    
            segment_outputs.append(x)
    
        pois = 0.15
        x_i = Concatenate()(segment_outputs)
        x_i = Dense(self.input_dim)(x_i)
        x_i = LeakyReLU(lk)(x_i)
        x_i = PoisonNoise(lam=pois, std=1)(x_i)
        x_i = Translation(trans)(x_i)
    
        x_f = Dense(self.input_dim, activation='tanh')(x_i)
        x_f = x_f * mult_scalar
    
        model = tf.keras.Model(entrada, x_f)
        model.summary()
        return model

    def make_discriminator_model_0(self):
        n=5
        step=round(self.input_dim/n) #devuelve entero
        lk = 0.3
        
        entrada = Input(shape=(self.input_dim))
        x = Dense(self.input_dim, input_dim = self.input_dim)(entrada)
        x = LeakyReLU(lk)(x)
        x = Dropout(0.5)(x)
        
        for i in range(1,n):
            if i == 3:
                x = MinibatchDiscrimination(num_kernel=100, dim_kernel=5)(x)
            else:
                x = Dense((n-i)*step)(x)
                x = LeakyReLU(lk)(x)
                x = Dropout(0.5)(x)
        
        y = Dense(self.input_dim, input_dim = self.input_dim)(entrada)
        y = LeakyReLU(lk)(y)
        y = Dropout(0.5)(y)
        
        for i in range(1,n):
            if i == 3:
                y = MinibatchDiscrimination(num_kernel=150, dim_kernel=8)(y)
            else:
                y = Dense((n-i)*step)(y)
                y = LeakyReLU(lk)(y)
                y = Dropout(0.5)(y)
            
        z = Concatenate(axis=1)([x,y])
        z = Flatten()(z)
        z = Dense(1, activation='sigmoid')(z)
        model = tf.keras.Model(entrada,z)
        return model
    
    def make_discriminator_model_1(self):
        n=5
        step=round(self.input_dim/n) #devuelve entero
        lk = 0.3
        
        entrada = Input(shape=(self.input_dim))
        x = Dense(self.input_dim, input_dim = self.input_dim)(entrada)
        x = LeakyReLU(lk)(x)
        x = Dropout(0.5)(x)
        
        for i in range(1,n):
            if i == 3:
                x = MinibatchDiscrimination(num_kernel=100, dim_kernel=5)(x)
            else:
                x = Dense((n-i)*step)(x)
                x = LeakyReLU(lk)(x)
                x = Dropout(0.5)(x)
        
        y = Dense(self.input_dim, input_dim = self.input_dim)(entrada)
        y = LeakyReLU(lk)(y)
        y = Dropout(0.5)(y)
        
        for i in range(1,n):
            if i == 3:
                y = MinibatchDiscrimination(num_kernel=150, dim_kernel=8)(y)
            else:
                y = Dense((n-i)*step)(y)
                y = LeakyReLU(lk)(y)
                y = Dropout(0.5)(y)
            
        z = Concatenate(axis=1)([x,y])
        z = Flatten()(z)
        z = Dense(1, activation='sigmoid')(z)
        model = tf.keras.Model(entrada,z)
        return model

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step_1(self,images):
        #Regular training of generator and discriminator. 
        noise = tf.random.normal([self.BATCH_SIZE, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
    
            real_output = self.discriminator_0(images, training=True)
            fake_output = self.discriminator_0(generated_images, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_0.trainable_variables)
    
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_0.trainable_variables))
        return gen_loss, disc_loss
    
    @tf.function
    def train_step_2(self,images):
        #Training with translations.
        noise = tf.random.normal([self.BATCH_SIZE, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            #Apply the traslations.
            trans = tf.random.uniform(shape=(), minval=1, maxval=11, dtype=tf.int32) #traslacion que queramos hacer
            generated_images_trans = self.translation(generated_images,trans)
            images_trans = self.translation(images,trans)
            
            real_output = self.discriminator_1(images_trans, training=True)
            fake_output = self.discriminator_1(generated_images_trans, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_1.trainable_variables)
    
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_1.trainable_variables))
        return gen_loss, disc_loss
    
    @tf.function
    def train_step_3(self,images):
        #2nd Training of discriminator.
        noise = tf.random.normal([self.BATCH_SIZE, self.latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=False)
            
            real_output = self.discriminator_0(images, training=True)
            fake_output = self.discriminator_0(generated_images, training=True)
    
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_0.trainable_variables)
    
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_0.trainable_variables))
        
        return gen_loss, disc_loss
    
    def train(self, epochs, stopping_criterias):
        
        data = self.spectra
        #Pon los datos en forma de tensor:
        data = tf.convert_to_tensor(data);
        seed = tf.random.normal([self.num_examples_to_generate, self.latent_dim])
        gif = []; losses = []
        evaluation = self.BATCH_SIZE
        shape_eval = tf.random.uniform((evaluation,))
        
        ##--------------------------------------##
        # Soft labels to evaluate fake & real data:
        y_real_eval = tf.ones_like((shape_eval))-0.1
        y_fake_eval = tf.zeros_like((shape_eval))
        ##--------------------------------------##
        
        #Stopping Criterias:
        fid_tol = stopping_criterias['FID_tol'];
        correlation_tol = stopping_criterias['correlation_tol'];
        cos_tol = stopping_criterias['cosine_tol'];
        
        for epoch in range(0,epochs):
            start = time.time()
            dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
            
            for image_batch in dataset: #estandar sin más
                gen_loss_1,dis_loss_1 = self.train_step_1(image_batch)
                
            for image_batch in dataset: #translaciones
                gen_loss_2,dis_loss_2 = self.train_step_2(image_batch) 
                
            for image_batch in dataset: #entrenamos el discriminador_0 2 veces
                gen_loss_3,dis_loss_3 = self.train_step_3(image_batch)
                
            #for image_batch in dataset: #training with noise
            #    self.train_step_4(image_batch)
            print(gen_loss_1,dis_loss_1,'\n')
            print(gen_loss_2,dis_loss_2,'\n')
            print(gen_loss_3,dis_loss_3,'\n')
            losses.append([epoch,gen_loss_1.numpy(),dis_loss_1.numpy(),gen_loss_2.numpy(),dis_loss_2.numpy(),gen_loss_3.numpy(),dis_loss_3.numpy()])

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator,epoch+1,seed)
            
            idx = tf.random.uniform((evaluation,),0,data.get_shape()[0],dtype=tf.int32)
            real_imgs = tf.gather(data,idx.numpy())
            noise_eval =  tf.random.normal((evaluation, self.latent_dim),mean=0.,stddev=1.)
            generated_imgs = self.generator.predict(noise_eval)

            #FID distance: 
            fid_value = self.calculate_fid(real_imgs.numpy(),generated_imgs)
            #Correlation coeficient: 
            correlation_value = self.calculate_correlation(real_imgs.numpy(),generated_imgs)
            #Cosine distance:
            cosine_distance = self.calculate_cos_dist(real_imgs.numpy(),generated_imgs)

            gif.append([epoch,fid_value,correlation_value,cosine_distance])
            print('Number of epochs:',epoch,'\n')
            print('Frechet Inception Distance:',fid_value,'\n')
            print("Correlation: ", correlation_value,'\n')
            print("Cosine Distance", cosine_distance,'\n')

            if fid_value <= fid_tol and correlation_value >= correlation_tol and cosine_distance <= cos_tol:# and epoch>200:
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
                return (self.discriminator_0,self.generator,gif,losses)
                        
        print('After ',epoch,' epochs the training function has been not able to find any proper GAN model')
        return None