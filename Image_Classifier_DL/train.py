# Imports here
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import models,datasets, transforms
from collections import OrderedDict

import argparse
import datetime

# Define all Methods to be used    

# Method to Pre-process and load training, validation, and testing data
def data_load(data_dir):
    # initialize the paths for training, validation and testing data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # transforms for training , validation & test data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_transforms  = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    # Load the training, validation, and testing datasets with ImageFolder
    #image_datasets

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data =  datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data =  datasets.ImageFolder(test_dir, transform=valid_transforms)

    #Using the image datasets and the transforms to define the dataloaders for training, validation, and testing
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    print ("Number of Classes = " + str(len(train_data.classes))) 
    print ("Training Sample size = " + str(len(train_data.samples)))
    print ("Validation Sample size = " + str(len(valid_data.samples)))
    #print ("Testing Sample size = " + str(len(test_data.samples)))
    print ("Testing Sample size = " + str(len(testloader.dataset)))
    return trainloader,validloader,testloader,train_data


#  Method to load pretrained Model to be used
def load_model_arch(arch):
     if arch == "vgg16":
        model = models.vgg16(pretrained=True)
     elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
     else:
        print("Model Architecture not recognized - currently supports only vgg16 or vgg19")
    # Freeze parameters so we don't backprop through them
     for param in model.parameters():
        param.requires_grad = False
    
     return model

# Method to load the classifier
def load_classifier(hidden_units):
    # Initialize the parameters
    input_size = 25088
    output_size = 102
    hidden_sizes = hidden_units
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                        ('relu1', nn.ReLU()),
                        ('drop1', nn.Dropout(p=0.5)),
                        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                        ('relu2', nn.ReLU()),
                        ('fc3', nn.Linear(hidden_sizes[1], output_size)),
                        ('output', nn.LogSoftmax(dim=1))]))
    return classifier


# Method for finding loss and accuracy of model on validation/test data
# depending on whether it is passed Validation data or Test data
def validation(model, valid_test_loader, criterion, gpu):
    valid_test_correct = 0
    valid_test_loss = 0

    for images, labels in (iter(valid_test_loader)):
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        # Forward pass
        output = model.forward(images)
        # loss
        loss = criterion(output, labels)
        valid_test_loss += loss.item()
        # Accuracy
        predicted_val = torch.max(output.data, 1)[1]
        valid_test_correct += (predicted_val == labels).sum().item()
        
    avg_valid_test_loss = valid_test_loss/len(valid_test_loader.dataset) # Validation/Test loss
    valid_test_acc = (100 * valid_test_correct)/len(valid_test_loader.dataset) # Validation/Test Accuracy %
    
    return avg_valid_test_loss, valid_test_acc


# Method to train the model
def train(model, trainloader, criterion,epochs,optimizer, validloader, gpu):
    train_loss = 0
    train_acc = 0
    steps = 0
    print_every = 36 # Define Training Batch size

    for e in range(epochs):
        start = datetime.datetime.now() # Set the Start time
        model.train()
        for images, labels in iter(trainloader):
            steps += 1
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            # clear gradients because gradients are accumalated
            optimizer.zero_grad()

            # Forward pass , Calculate loss ,backward passes and update weights
            output = model.forward(images)       
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # Training Loss
            train_loss += loss.item() 

            # Training Accuracy
            predict_train = torch.max(output.data, 1)[1]
            train_acc += (predict_train == labels).sum().item()

            if(steps % print_every == 0): 
                # perform validation - ensure network is in evaluation mode for inference
                model.eval()
                # Turn off gradients for validation to save computation time and memory
                with torch.no_grad():
                # get the loss and accuracy on validation dataset
                    valid_loss,valid_acc = validation(model, validloader, criterion,gpu)
                    print("Epoch:{}/{} ".format(e + 1, epochs),
                              "Training Loss: {:.4f} ".format(train_loss/(print_every*trainloader.batch_size)),
                              "Training Accuracy: {:.2f}% ".format((100*train_acc)/(print_every*trainloader.batch_size)),
                              "Validation Loss: {:.4f} ".format(valid_loss),
                              "Validation Accuracy: {:.2f}%".format(valid_acc))

                train_loss = 0
                train_acc = 0
                model.train()

        # End of epoch
        stop = datetime.datetime.now()
        total_time = stop - start
        print('Total time spent in training the model for this epoch : {:.2f} mins'.format(total_time.total_seconds()/60))
        print('-----------------------------------------------------------------------------------------------------------------')

        
# Method to perform testing of the model on the test data set
def test(model,testloader,criterion,gpu):
    # Initialize
    test_loss = 0
    test_acc = 0
    start = datetime.datetime.now() # Capture the Start time

    # Call validation function for testing performance on Test Set
    test_loss,test_acc = validation(model, testloader, criterion,gpu)
    stop = datetime.datetime.now() # Capture the Stop time
    print ("Test Set size = " + str(len(testloader.dataset)))
    print ("Testing Loss: {:.4f} ".format(test_loss))
    print ("Testing Accuracy: {:.2f}%".format(test_acc))

    total_time = stop - start
    print('Total time spent on testing : {:.2f} mins'.format(total_time.total_seconds()/60))


# Method to save the trained model to a file so that it could be reused later
def save_model(model, filepath, arch,criterion, optimizer, train_data, epochs,lr):
    model.class_to_idx = train_data.class_to_idx
    model.cpu() # Move model to CPU
    model_dict = {
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'criterion_state': criterion.state_dict(),
                    'classifier': model.classifier,
                    'class_to_idx': model.class_to_idx,
                    'learn_rate' : lr,
                    'epoch': epochs
                }
    # Save the model
    torch.save(model_dict,filepath)
    print("Trained model saved on file :" + filepath)
   

    
# Main Program - Execution starts here 
def main():
    # Setting up the command line parser to define the options
    parser = argparse.ArgumentParser(description = 'Train a Flower classifier')
    parser.add_argument('--data_dir', type=str, default='flowers', required=True, help='location of dataset directory')
    parser.add_argument('--save_dir', type=str, default='model_state_chkpoint.pth', help='file path for saving model checkpoint')
    parser.add_argument('--arch' , type=str, default= 'vgg19',help = 'Select Pretained neural network model')
    parser.add_argument('--hidden_units', type=int, nargs='+' , default=[1024,256], help='hidden units for fc layer')
    parser.add_argument('--gpu' , type=bool, default=False, help = 'Use GPU or not')
    parser.add_argument('--lr' , type=float, default= 0.001, help = 'Learning Rate')
    parser.add_argument('--epochs' , type=int, default=10 , help = 'Number of Epochs')
    # Load the command line arguments entered by the user
    args = parser.parse_args()
    
    print()
    print('-------------------------Image Classifier-------------------------------')
    print()
    print('Loading and processing data....................')
    print()
    #pre-process data
    train_loader, valid_loader, test_loader,train_data = data_load(args.data_dir)
    print()
    print('Loading pre-trained Model......................')
    print()
    # Load pre-trained model
    model = load_model_arch(args.arch)

    # Define Classifier that is compatible with the desired output
    classifier = load_classifier(args.hidden_units)

    # replace the pre-trained classifier of the chosen model with the above one
    model.classifier = classifier
    # Move model to GPU if GPU is available and argument to use GPU is true
    is_gpu = torch.cuda.is_available() # check if GPU facility available
    if args.gpu: # if --gpu True option used in command-line
        if is_gpu: #Use GPU for computations if available
            model = model.to('cuda') 
            print ("Using GPU to speed up processing")
        else:
            print("GPU is not available - so using CPU")
    print()      
    # Define optimizer and loss function
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    
    print('Training the classifier........................')
    # Train the model classifier 
    train(model,train_loader, criterion,args.epochs, optimizer, valid_loader,args.gpu)

    print()
    print('Testing the trained model ........................')
    # Test
    test(model, test_loader, criterion, args.gpu)
    
    print()
    print('Saving the trained model .........................')
    # Save Model to a file
    filepath = args.save_dir
    save_model(model, filepath, args.arch, criterion, optimizer, train_data, args.epochs, args.lr)
          
if __name__ == "__main__":
    main()
