'''
Created on 27/12/2018

@author: johan mejia
'''

class intersection_over_union: # Calculate the IoU between two boxes with normalize measure
    # Construct
    def __init__(self,boxA = [0.0,0.0,1.0,1.0], boxB = [0.0,0.0,1.0,1.0]): # defaultBoxes = [xmin,ymin,xmax,ymax]
        self.__boxA = boxA # First Attribute
        self.__boxB = boxB # Second Attribute
        self.__iou = 0.0 # Default IoU
        self.IoUestimate()
    
    # Functions
    def setBoxA(self,boxA):
        self.__boxA = boxA
    
    def setBoxB(self,boxB):
        self.__boxB = boxB
    
    def gerIoU(self):
        return self._iou

    def IoUestimate(self, boxA = None, boxB = None):
        if boxA is not None: self.setBoxA(boxA)
        if boxB is not None: self.setBoxB(boxB)

        # Proof boxes measures
        if len(self.__boxA) != 2 and len(self.__boxA) != 4:
            print("Invalid measures boxA")
            return 0
        if len(self.__boxB) != 2 and len(self.__boxB) != 4:
            print("Invalid measures boxB")
            return 0
        
        # [w,h] to [xmin,ymin,xmax,ymax] with center (0.5, 0.5)
        if len(self.__boxA) == 2: # boxA 
            self.__boxA = [0.5*(1-self.__boxA[0]),0.5*(1-self.__boxA[1]),0.5*(1+self.__boxA[0]),0.5*(1+self.__boxA[1])]
        if len(self.__boxB) == 2: # boxB
            self.__boxB = [0.5*(1-self.__boxB[0]),0.5*(1-self.__boxB[1]),0.5*(1+self.__boxB[0]),0.5*(1+self.__boxB[1])]
        
        # Determine the (x,y) - coordinates of the intersection rectangle
        xA = max(self.__boxA[0], self.__boxB[0])
        xB = min(self.__boxA[2], self.__boxB[2])
        yA = max(self.__boxA[1], self.__boxB[1])
        yB = min(self.__boxA[3], self.__boxB[3])
     
        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
     
        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (self.__boxA[2] - self.__boxA[0]) * (self.__boxA[3] - self.__boxA[1])
        boxBArea = (self.__boxB[2] - self.__boxB[0]) * (self.__boxB[3] - self.__boxB[1])
     
        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        self._iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return self._iou