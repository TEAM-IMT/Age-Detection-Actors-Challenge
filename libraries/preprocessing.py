## Libraries ###########################################################
import cv2, os, sys, tqdm
import numpy as np
import matplotlib.pyplot as plt

## Functions and Class #################################################
class Preprocessing:
    # Constructor
    def __init__(self, gray_scale = False, resize = (None, None), normalize = False):
        self.gray_scale = gray_scale
        self.resize = resize
        self.normalize = normalize

    # Methods
    def image_read(self, image_file, process = True): 
        image = cv2.imread(image_file)[...,::-1] # RGB Format
        if process: image = self.processing(image)
        return image 
    
    def processing(self, image):
        # Gray scale
        if self.gray_scale: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize
        rew, reh = self.resize
        if rew is not None and reh is None: reh = image.shape[0]
        elif rew is None and reh is not None: rew = image.shape[1]
        if rew is not None and reh is not None: 
            image = cv2.resize(image, (rew, reh), interpolation = cv2.INTER_LANCZOS4)
        
        # Normalize
        norm_img = np.zeros((image.shape[1], image.shape[0]))
        if self.normalize: image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)/255.0
        return image

    def images_processing(self, image_list): # Directory path or image files in list
        images = []
        print("[INFO] Reading images...")
        if type(image_list) == str and os.path.isdir(image_list):
            image_list = [os.path.join(image_list, f) for f in os.listdir(image_list)]
        for ifile in tqdm.tqdm(image_list):
            if os.path.isfile(ifile):
                images.append(self.image_read(ifile)[None])
            else:
                print("[WARNING] {} invalid. Ignored it.".format(ifile))
        return np.vstack(images)

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

if __name__ == "__main__":
    if len(sys.argv) != 2 :
        sys.exit("[ERROR] Usage: python3" + sys.argv[0] + "<path_image_or_dirimage>")
    
    ipath = sys.argv[1]
    preproc = Preprocessing(gray_scale = False, resize = (300,300), normalize = True)

    # Process
    if os.path.isfile(ipath):
        image = preproc.image_read(ipath, process = False)
        image_pro = preproc.image_read(ipath, process = True)
        print("[INFO] Shape image processed: ", image_pro.shape)
        print("[INFO] Image values between [{},{}].".format(np.min(image_pro), np.max(image_pro)))

        # Plot results
        image_plot(image, image_pro)
    elif os.path.isdir(ipath):
        images = preproc.images_processing(ipath)
        print("[INFO] Batch image shape: ", images.shape)
        images_plot(images, nrows = 2, ncols = 5)
    else:
        print("[ERROR] Invalid path.")