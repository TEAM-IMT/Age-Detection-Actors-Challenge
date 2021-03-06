## Libraries ###########################################################
import cv2, os, sys, tqdm, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import dlib
except:
    sys.exit("[ERROR] ModuleNotFoundError: No module named 'dlib'. Also, you need install cmake")
from Alignment import FaceAligner
from help import rect_to_bb

from autoencoder import DB_VAE

## Functions and Class #################################################
class Preprocessing:
    # Constructor
    def __init__(self, gray_scale = False, resize = (None, None), normalize = False, rotate = None, vae_weighs_path = None):
        self.gray_scale = gray_scale
        self.resize = resize
        self.normalize = normalize
        self.face_alig = dlib.shape_predictor(rotate) if rotate is not None else None
        self.face_alig = FaceAligner(self.face_alig, desiredFaceWidth = 256) if rotate is not None else None
        self.vae_weighs_path = vae_weighs_path
        if self.vae_weighs_path is not None: # Values to default if we apply VAE
            self.vae_model = DB_VAE(400) # Latent space
            self.vae_model.load(self.vae_weighs_path)
            self.gray_scale = False
            self.resize = (64,64)

    # Methods
    def _rotation(self, image, image_grey = None):
        (x, y, w, h) = (0, 0, image.shape[1], image.shape[0])
        rect = dlib.rectangle(left = x, top = y, right = w, bottom = h)
        if image_grey is None: image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = self.face_alig.align(image, image_grey, rect)      
        image = np.delete(image, np.where(~image.any(axis = 0))[0], axis = 1) # Remove zero-pad border
        return np.delete(image, np.where(~image.any(axis = 1))[0], axis = 0)

    def image_read(self, image_file, process = True):
        image = cv2.imread(image_file)[...,::-1] # RGB Format
        if process: image = self.processing(image)
        return image 
    
    def processing(self, image):
        # Gray scale
        if self.gray_scale: 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Rotation
        if self.face_alig is not None:
            image = self._rotation(image, image_grey = image.copy() if self.gray_scale else None)

        # Resize
        rew, reh = self.resize
        if rew is not None and reh is None: reh = image.shape[0]
        elif rew is None and reh is not None: rew = image.shape[1]
        if rew is not None and reh is not None: 
            image = cv2.resize(image, (rew, reh), interpolation = cv2.INTER_LANCZOS4)
        
        # Normalize
        if self.normalize: 
            norm_img = np.zeros((image.shape[1], image.shape[0]))
            image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

        # VAE codification
        if self.vae_weighs_path is not None:
            image = self.vae_model.image_encode(image[None]/255.0)[0] # Need to image normalize 
        return image

    def images_processing(self, image_list): # Directory path or image files in list
        images = []
        print("[INFO] Reading images...")
        if type(image_list) == str and os.path.isdir(image_list):
            image_list = [os.path.join(image_list, f) for f in os.listdir(image_list)]
        for ifile in tqdm.tqdm(image_list):
            if os.path.isfile(ifile): images.append(self.image_read(ifile))
            else: print("[WARNING] {} invalid. Ignored it.".format(ifile))
        return np.stack(images, axis = 0)

    def images_decode(self, images):
        return self.vae_model.image_decode(images)

def image_plot(image_org, image_pro):
    _, axs = plt.subplots(ncols = 2, figsize = (10, 5))
    canvas = {0: {'title': "Original image", 'image': image_org},
        1: {'title': "Image processed", 'image': image_pro}
    }
    
    for i, ax in enumerate(axs):
        ax.imshow(canvas[i]['image'], cmap = 'gray' if len(canvas[i]['image'].shape) == 2 else None)
        ax.set_title(canvas[i]['title'])
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def images_plot(images_batch, nrows = 2, ncols = 3):
    _, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10, 5))
    iind = np.random.randint(0, len(images_batch), size = nrows*ncols)
    for i, ax in enumerate(axs.reshape(-1)):
        image = images_batch[iind[i]]
        ax.set_title("Image " + str(iind[i]))
        ax.imshow(image, cmap = 'gray' if len(image.shape) == 2 else None)
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()


## Main ################################################################
if __name__ == "__main__":
    if len(sys.argv) != 2 :
        sys.exit("[ERROR] Usage: python3 " + sys.argv[0] + " <path_image_or_dirimage>")
    
    split_data = False
    train, test, valid = 0.7, 0.15, 0.15
    ipath = sys.argv[1]
    preproc = Preprocessing(gray_scale = False, resize = (106,128), normalize = True, # Best (80,100)
                # rotate = "./shape_predictor_68_face_landmarks.dat",
                vae_weighs_path = './weights/vae_model.h5') # Ignore all parameters

    # Process
    if os.path.isfile(ipath):
        image = preproc.image_read(ipath, process = False)
        image_pro = preproc.image_read(ipath, process = True)
        print("[INFO] Shape image and type processed: ", image_pro.shape, type(image_pro), image_pro.dtype)
        print("[INFO] Image values between [{},{}].".format(np.min(image_pro), np.max(image_pro)))

        # Plot results
        if preproc.vae_weighs_path is not None: image_plot(image, preproc.images_decode(image_pro[None])[0])
        else: image_plot(image, image_pro)
    elif os.path.isdir(ipath):
        prefix = 'color_' if not preproc.gray_scale else 'gray_'
        if split_data:
            table = pd.read_csv('./data/Train_Augmented.csv', sep = ',', header = 0, index_col = 0)
            print("[INFO] Total data: ", len(table))
            data = {cl: table.loc[table['Class'] == cl, 'ID'].to_list() for cl in table['Class'].unique()}
            print("[INFO] After shuffle: ", ["{}: {}".format(key, data[key][:3]) for key in data.keys()])
            for key in data.keys(): np.random.shuffle(data[key]) # Shuffle
            print("[INFO] After shuffle: ", ["{}: {}".format(key, data[key][:3]) for key in data.keys()])
            N = {k:len(v) for k,v in data.items()}
            data = {'train': sum([data[k][:int(train*N[k])] for k in data.keys()], []),
                    'valid': sum([data[k][int(train*N[k]):int((train+valid)*N[k])] for k in data.keys()], []),
                    'test': sum([data[k][int((train+valid)*N[k]):] for k in data.keys()], [])}
            for key in data.keys(): np.random.shuffle(data[key]) # Last shuffle
            print("[INFO] After shuffle: ", ["{}: {}".format(key, data[key][:3]) for key in data.keys()])
            print("[INFO] Lenght of sets: ", ["{}: {}".format(key, len(data[key])) for key in data.keys()])
            table = table.set_index('ID')
            for key, value in data.items():
                print("[INFO] {} process".format(key))
                images = preproc.images_processing([os.path.join(ipath,x) for x in value])
                labels = table.loc[value, 'Class'].values
                print("[INFO] Batch image shape: ", images.shape)
                pickle.dump([images, labels], open(prefix + key + ".pkl","wb"))
        else:
            image_list = os.listdir(ipath)
            images = preproc.images_processing([os.path.join(ipath,x) for x in image_list])
            pickle.dump([images, image_list], open(prefix + "test_unlabel.pkl","wb"))
        if preproc.vae_weighs_path is None: images_plot(images[:1000], nrows = 3, ncols = 8)
        else: images_plot(preproc.images_decode(images[:1000]), nrows = 3, ncols = 8)
    else:
        print("[ERROR] Invalid path.")
