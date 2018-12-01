# Imports

import numpy as np
import torch
from torch import nn
from torchvision import models
import json
import argparse
import datetime
from PIL import Image

# Define Methods to be used

# Method to pre-process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Float tensor
    '''
    # Process a PIL image for use in a PyTorch model
    
    # Open the image
    img = Image.open(image)
    img_wid = img.size[0]  # image width
    img_ht  = img.size[1]  # image height
    
    # Resize image inplace using thumbnail() - automatically maintains aspect ratio of original image
    if (img_wid > img_ht): 
        img.thumbnail((img_wid, 256)) 
    else :
        img.thumbnail((256, img_ht)) 
       
    # Center Crop image to get dimensions- 224 x 224 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,top_margin)) 
    
    # Normalize image
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) # mean
    std = np.array([0.229, 0.224, 0.225]) # standard deviation
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
   
    # convert numpy array to a float tensor before return
    return torch.from_numpy(img).type(torch.FloatTensor) 


# function to load the model at last known checkpoint and rebuild the model if required
def load_model(filepath):
    model_data = torch.load(filepath,map_location=lambda storage, loc: storage)
    if model_data['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_data['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Invalid Model Architecture - can use only vgg16 or vgg19")
        return   
    model.classifier = model_data['classifier']
    model.class_to_idx = model_data['class_to_idx']
    model.load_state_dict(model_data['state_dict'])
       
    return model;


# Method to Predict the top k possible classes of the given image of a flower and their probabilities
def predict (image, model,topk,cat_names,gpu):
    ''' Predict the top 'k' most likely class (or classes) of an image using the trained deep learning model.
    '''
    # process image file
    if gpu:
        img_tensor = process_image(image).to('cuda')
    else:
        img_tensor = process_image(image)
    #https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/3
    img_tensor.unsqueeze_(0) # adds dimension of size 1 at 0th index of image tensor - this becomes the batch size parameter
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model.forward(img_tensor))
        probs, indices = torch.topk(ps, topk)
    
        # convert returned indices to get corresponding classes
        indices = np.array(indices)            
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in indices[0]]
        
        # load in a mapping from categorylabel to category name
        with open(cat_names, 'r') as f:
            cat_to_name = json.load(f)
       
        # map the class name with collected topk classes
        classlist = []
        for cls in classes:
            classlist.append(cat_to_name[str(cls)])
                    
        return probs.to('cpu'), classlist
    



# Code Execution starts here
def main():
 # Setting up the command line parser to define the options
    parser = argparse.ArgumentParser(description = 'Predict Class of the Flower')
    parser.add_argument('--image_path', type=str, required=True, help='Path of image file')
    parser.add_argument('--topk', type=int, default=5, help='Show top k probabilities')
    parser.add_argument('--saved_model' , type=str, default='model_state.pth', help='File Path of Saved model')
    parser.add_argument('--gpu' , type=bool, default=False, help = 'Use GPU or not')
    parser.add_argument('--category_names' , type=str, default='cat_to_name.json', help='File Path of your JSON mapper - category to name')
    # Load the command line arguments entered by the user
    args = parser.parse_args()
    
      
    # Load Model
    #print(args.saved_model)
    model = load_model(args.saved_model)
    is_gpu = torch.cuda.is_available() # check if GPU facility available
    if args.gpu: # if --gpu True option used in command-line
        if is_gpu: #Use GPU for computations if available
            model = model.to('cuda') 
            print("Using GPU to speed up processing")
        else:
            print("GPU is not available - so using CPU")
    else:
        print("Using CPU.....")
        
    # Predict the top k most likely classnames and their corresponding probabilities
    class_probs, class_names = predict(args.image_path,model,args.topk,args.category_names,args.gpu)
    print()
    print('--------------Prediction for the given image------------------')
    print()
    print("> {}  with a probability of {:.3f}".format(str(class_names[0]), float(class_probs[0][0].numpy())))
    print()
    print("----------Top k possible Classes with Probabilities-----------")
    print()
    for idx in range(0 , args.topk):
        print("#{} {} with a probability of {:.3f}".format(idx+1, str(class_names[idx]), float(class_probs[0][idx].numpy()))) 
    print()  
    print('--------------------------------------------------------------')

if __name__ == "__main__":
    main()
