from torch.utils.data import Dataset
import cv2


class CustomDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    '''
    def __init__(self, labels_csv, transformation, pretrained):

        self.labels_csv = labels_csv

        self.images_path = self.labels_csv["Path"].to_list()

        
        self.transformation = transformation

        self.pretrained = pretrained
        
    
    def __getitem__(self, idx):
        
        img_path = self.labels_csv.loc[idx, 'Path']



        img = self.load_and_preprocess(img_path)

        sample = {
            "image": img,
            "label": int(self.labels_csv.loc[idx, 'Label']),
            "id":   self.labels_csv.loc[idx, 'ID']   

        }
        

        return sample
        
    def load_and_preprocess(self, img_path):

        if self.pretrained:
            image = cv2.imread(img_path)
        else:
            image = cv2.imread(img_path)

        
        if self.transformation:
            image = self.transformation(image=image)
            return image['image']
        else:
            return image
    
        
    
    def __len__(self):
 
        return len(self.labels_csv)