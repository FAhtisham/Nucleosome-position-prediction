from pycm import *
import torch
import torch.nn as nn

import config
import logging

from utils import create_embd_matrix, load_pretrained_embd

logger = config.logging.getLogger('train')
fh= logging.FileHandler(config.log_file)
logger.addHandler(fh)

def reset_weights(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.Conv1d):
    m.reset_parameters()


def train(fold, train_loader, epoch, model, optimizer, loss_function):
  model.train()

  total_loss = 0.0
  correct = 0.0
  total = 0.0
  for i,x in enumerate(train_loader):
    
    samples, labels = x
    # logger.info("sample ", samples.size())

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

  logger.info("Epoch {}| Train Loss {}| Overall Train Accuracy {}| Actual Train Accuracy {} ".format(epoch, total_loss/len(train_loader), (100*(correct/total)), actual_accuracy))

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
  
  logger.info(" Val fold {}| Val  Loss {}| Overall Val Accuracy {}| Actual Val Accuracy {}".format(fold, total_loss/len(val_loader), (100*(correct/total)), actual_accuracy/len(val_loader)))
  return actual_accuracy/len(val_loader)

  
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

  logger.info("Overall Test Accuracy {}| Actual Test Accuracy {} ".format((100*(correct/total)), total_actual_accuracy/len(test_loader)))
  return total_actual_accuracy/len(test_loader)


  
