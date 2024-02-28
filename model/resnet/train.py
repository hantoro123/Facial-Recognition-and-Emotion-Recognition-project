import torch
from tqdm import tqdm


def do_train(model, trainloader, criterion, lr=0.001, training_epochs=10, device="cuda", step=3):

    criterion = criterion.to(device) # Softmax is internally computed.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if step == 0 :
        pass
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)

    print("Training start !")
    
    #list of loss
    train_losses = []
    train_acc = []
    model.to(device)
   
    for epoch in range(training_epochs):  # loop over the dataset multiple times
        
        model.train()
        total = 0.0
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainloader): 
            # get the inputs
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # set gradients to zero

            # forward + backward + optimize
            outputs = model(inputs) # forward myNN
            loss = criterion(outputs, labels) #compute the loss
            loss.backward() #back prop
            optimizer.step() #weight update
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            running_loss += loss.item()/len(trainloader)
            running_corrects += torch.sum(predicted == labels.data).item()
            
        train_losses.append(running_loss)
        trn_acc = running_corrects/total
        train_acc.append(trn_acc)
        lr_scheduler.step()
        
    return train_losses, train_acc, model

def do_evaluate(model, testloader, device, criterion):
    correct = 0
    total = 0
    validation_loss = 0.
    criterion = criterion.to(device) # Softmax is internally computed.
    
    val_loss_list = []
    val_acc = []
    model.to(device)

    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_loss = criterion(outputs, labels)
            validation_loss += val_loss.item()/len(testloader)
                            
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    validation_acc = correct/total

    val_loss_list.append(validation_loss)
    val_acc.append(validation_acc)
    
    return val_loss_list, val_acc