from pycm import *
import torch
import torch.nn as nn


import config 

def reset_weights(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
    m.reset_parameters()


def train(fold, train_loader, epoch, model, optimizer, loss_function):
  model.train()

  total_loss = 0.0
  correct = 0.0
  total = 0.0
  
  for i,x in enumerate(train_loader):
    
    samples, labels = x

    predicted = model(samples.cuda())
    optimizer.zero_grad()
    loss = loss_function(predicted, labels.cuda())

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total += labels.size(0)
    
    predicted = torch.round(predicted)
    correct += (predicted.cpu() == labels).sum().item()

    # total_loss_e = total_loss
  cm = ConfusionMatrix(actual_vector = labels.cpu().numpy().tolist(), predict_vector = predicted.cpu().detach().numpy().tolist())
  actual_accuracy = (sum(cm.TP.values())+ sum(cm.TN.values()))/(sum(cm.TP.values())+sum(cm.TN.values())+sum(cm.FP.values())+sum(cm.FN.values())) 

  print("Epoch {}| Train Loss {}| Overall Train Accuracy {}| Actual Train Accuracy {} ".format(epoch, total_loss/len(train_loader), (100*(correct/total)), actual_accuracy), file=open('log.txt', 'a'))

def validate(fold, val_loader, model, optimizer, loss_function):
  model.eval()
  total_loss = 0.0
  correct = 0.0
  total = 0.0
  actual_accuracy =0.0
  for i,x in enumerate(val_loader):
    samples, labels = x
    # samples = samples.cuda()
    predicted = model(samples.cuda())
    loss = loss_function(predicted, labels.cuda())
    total_loss += loss.item()
    total += labels.size(0)
    
    predicted = torch.round(predicted)
    correct += (predicted.cpu() == labels).sum().item()

    cm = ConfusionMatrix(actual_vector = labels.cpu().numpy().tolist(), predict_vector = predicted.cpu().detach().numpy().tolist())
    actual_accuracy += (sum(cm.TP.values())+ sum(cm.TN.values()))/(sum(cm.TP.values())+sum(cm.TN.values())+sum(cm.FP.values())+sum(cm.FN.values())) 
  print(" Val fold {}| Val  Loss {}| Overall Val Accuracy {}| Actual Val Accuracy {}".format(fold, total_loss/len(val_loader), (100*(correct/total)), actual_accuracy/len(val_loader)), file = open('log.txt', 'a'))


  
def test(test_loader, model):
  model.eval()
  total_loss = 0.0
  correct = 0.0
  total = 0.0
  total_actual_accuracy=0.0
  for i,x in enumerate(test_loader):
    samples, labels = x

    # samples = samples.cuda()
    predicted = model(samples.cuda())
    total += labels.size(0)    
    predicted = torch.round(predicted)
    correct += (predicted.cpu()== labels).sum().item()

    
    cm = ConfusionMatrix(actual_vector = labels.cpu().numpy().tolist(), predict_vector = predicted.cpu().detach().numpy().tolist())
    total_actual_accuracy += (sum(cm.TP.values())+ sum(cm.TN.values()))/(sum(cm.TP.values())+sum(cm.TN.values())+sum(cm.FP.values())+sum(cm.FN.values())) 

  print("Overall Test Accuracy {}| Actual Test Accuracy {} ".format((100*(correct/total)), total_actual_accuracy/len(test_loader)), file = open('log.txt', 'a'))


  



class TrainClass():
  def __init__(self, ae_list):
    super(TrainClass, self).__init__()
    
    loss_fn = nn.CrossEntropyLoss()
    self.ae_list = ae_list

  def TrainAll(self, train_DataLoader, input_embd, loss_fn):    
    for i in range(len(self.ae_list)):
      print("Model : {}".format(i+1), file=open("log.txt", "a"))
      self.ae_list[i] = self.Train_by(self.ae_list[:i+1], train_DataLoader, input_embd, loss_fn)
      torch.save(self.ae_list[i].state_dict(), config.data_name+"model"+ str(i) +".pt")    



    def Train(self, model):
      epochs  = 75
      for epoch in range(epochs):
        epoch_loss = 0.0
        for i,x in enumerate(train_DataLoader):
          if torch.cuda.is_available():
              x = x[0].cuda()
          else:
              x = x[0]
          out=model(input_embd(x))
          optimizer.zero_grad()
          loss = loss_fn(out.transpose(1,2).transpose(0,2), x)
          epoch_loss += loss.item()
          loss.backward()
          optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch+1,epoch_loss ))

        return model

  def Train_by(self, model_list, train_DataLoader, input_embd, loss_fn):
    optimizer = torch.optim.Adam(model_list[-1].parameters(), lr = 0.001)
    epochs  = 120
    for epoch in range(epochs):
      epoch_loss = 0.0
      for i,x in enumerate(train_DataLoader):
        if torch.cuda.is_available():
            x = x[0].cuda()
        else:
            x = x[0]

        if len(model_list)>1:
          embdd = input_embd(x)
          out = embdd
          for m in range(len(model_list)-1):
            with torch.no_grad():
              out, _=model_list[m].encoder_block(out)
        else:
          out=input_embd(x)
        
        out=model_list[-1](out)
        optimizer.zero_grad()
        
        loss = loss_fn(out.transpose(1,2).transpose(0,2), x)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
      print("Epoch: {}, Loss: {}".format(epoch+1,epoch_loss/len(train_DataLoader) ), file=open("log.txt", "a"))

    return model_list[-1]