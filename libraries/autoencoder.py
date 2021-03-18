## Libraries ###########################################################
import functools, cv2, sys
import tensorflow as tf
import matplotlib.pyplot as plt

## Functions and Class #################################################
class DB_VAE(tf.keras.Model):
    def __init__(self, latent_dim = 100, n_filters = 12):
        super(DB_VAE, self).__init__()
        # Save inputs
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        self.n_outputs = 2*self.latent_dim + 1 # [y_logit[0], zmean[1:latent_dim+1], zlogsig[latent_dim+1:]]

        self.encoder = self._make_standard_classifier()
        self.decoder = self._make_face_decoder_network()

    def encode(self, x):
        encoder_output = self.encoder(x)
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        z_mean = encoder_output[:, 1:self.latent_dim+1] 
        z_logsigma = encoder_output[:, self.latent_dim+1:]
        return y_logit, z_mean, z_logsigma
    
    def image_encode(self, x):
        return self.encoder(x).numpy()

    def decode(self, z):
        return self.decoder(z)
    
    def image_decode(self, x):
        z = self.reparameterize(x[:, 1:self.latent_dim+1], x[:, self.latent_dim+1:])
        return self.decode(z).numpy()

    # VAE reparameterization: given a mean and logsigma, sample latent variables ~ N(z_mean, z_logsigma)
    def reparameterize(self, z_mean, z_logsigma):
        batch, latent_dim = z_mean.shape
        epsilon = tf.random.normal(shape = (batch, latent_dim))
        return z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon

    # The call function will be used to pass inputs x through the core VAE
    def call(self, x): 
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        # Reparameterization and reconstruction
        z = self.reparameterize(z_mean, z_logsigma)
        recon = self.decode(z)
        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        return self.encode(x)[0] # y_logit

    def load(self, path):
        self(tf.random.uniform((1,64,64,3))) # Create connections
        self.load_weights(path)

    # Encoder networks
    def _make_standard_classifier(self):
        Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
        BatchNormalization = tf.keras.layers.BatchNormalization
        Flatten = tf.keras.layers.Flatten
        Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
        encoder = tf.keras.Sequential([
            Conv2D(filters=1*self.n_filters, kernel_size=5,  strides=2),
            BatchNormalization(),
            Conv2D(filters=2*self.n_filters, kernel_size=5,  strides=2),
            BatchNormalization(),
            Conv2D(filters=4*self.n_filters, kernel_size=3,  strides=2),
            BatchNormalization(),
            Conv2D(filters=6*self.n_filters, kernel_size=3,  strides=2),
            BatchNormalization(),
            Flatten(),
            Dense(512),
            Dense(self.n_outputs, activation=None),
        ])
        return encoder
    
    # Decoder networks
    def _make_face_decoder_network(self):
        Conv2DTranspose = functools.partial(tf.keras.layers.Conv2DTranspose, padding ='same', activation ='relu')
        BatchNormalization = tf.keras.layers.BatchNormalization
        Flatten = tf.keras.layers.Flatten
        Dense = functools.partial(tf.keras.layers.Dense, activation = 'relu')
        Reshape = tf.keras.layers.Reshape

        # Build the decoder network using the Sequential API
        decoder = tf.keras.Sequential([
            # Transform to pre-convolutional generation
            Dense(units=4*4*6*self.n_filters), # 4x4 feature maps (with 6N occurances)
            Reshape(target_shape=(4, 4, 6*self.n_filters)),
            Conv2DTranspose(filters=4*self.n_filters, kernel_size=3,  strides=2), # Upscaling convolutions (inverse of encoder)
            Conv2DTranspose(filters=2*self.n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=1*self.n_filters, kernel_size=5,  strides=2),
            Conv2DTranspose(filters=3, kernel_size=5,  strides=2),
        ])
        return decoder

## Main ################################################################
def plot_images(img_orig, img_pred):
    print(img_orig.shape, img_pred.shape)
    _, axs = plt.subplots(ncols = 2, figsize = (10, 5))
    canvas = {0: {'title': "Original image", 'image': img_orig},
        1: {'title': "Image VAE", 'image': img_pred}
    }
    
    for i, ax in enumerate(axs):
        ax.imshow(canvas[i]['image'], cmap = 'gray' if len(canvas[i]['image'].shape) == 2 else None)
        ax.set_title(canvas[i]['title'])
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def image_preprocess(image):
    image = cv2.resize(image, (64,64), interpolation = cv2.INTER_LANCZOS4) # Resize
    return image/255.0 # Image normalize

if __name__ == "__main__":
    latent_dim  = 100
    dbvae = DB_VAE(latent_dim)
    sample = tf.random.uniform((1,64,64,3)) # Networkt execute to create connections
    print("[INFO] Random prediction before load weights: ", dbvae.predict(sample))
    dbvae.load('./weights/vae_model.h5')
    print("[INFO] Random prediction after load weights: ", dbvae.predict(sample))

    if len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])[...,::-1] # RGB format
        image = image_preprocess(image) # Resize and normalize
        image_pred = dbvae.image_decode(dbvae.image_encode(image[None])).squeeze()
        plot_images(image, image_pred)
    elif len(sys.argv) > 2:
        sys.exit("[ERROR] Usage: python3 " + sys.argv[0] + " <path_image_or_dirimage>")
