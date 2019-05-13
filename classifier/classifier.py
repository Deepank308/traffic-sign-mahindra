import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pandas as pd 
import numpy as np
import cv2
import os
from torchvision import transforms, utils



class dataset(Dataset):
	def __init__(self, images_folder, annotations_file, transform):
		self.image_folder = images_folder
		annotations = pd.read_csv(annotations_file)
		self.annotations = np.array(annotations.iloc[:,:])
		self.transform = transform
        
	def __len__(self):
		return self.annotations.shape[0]

	def __getitem__(self, id):
		img_path = self.annotations[id,0]
		X = cv2.imread(self.image_folder + img_path)
		X = cv2.resize(X,(32, 32))
		# print(X.shape)
		y = np.zeros((1,7))
		y[0,self.annotations[id, 1]] = 1
		# print(y[0])
		if self.transform:
			X = self.transform(X)
		# print(X, y)
		X = np.array(X)
		X = X.transpose([2, 0, 1])
		# print(X.shape)
		return X, y[0]

class Net(nn.Module):
	def __init__(self, num_classes):
		super(Net, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,32,5,1,2),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(32),

			nn.Conv2d(32,64,5,1,2),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(64),

			nn.Conv2d(64,128,3,1,1),
			nn.ReLU(True),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(128),
			)
		self.in_features = 2048
		self.out_features = num_classes
		self.fc = nn.Linear(self.in_features, 256)
		self.dropout= nn.Dropout(0.5)
		self.classifier = nn.Linear(256, num_classes)


	def forward(self, X):
		output = self.features(X)
		output = output.view(output.shape[0], -1)
		output = self.fc(output)
		output = self.dropout(output)
		# print(output.shape)
		output = self.classifier(output)
		# output = F.softmax(output)
		return output

def train(net, data_loader, optimizer, batch_size):
	epoch_loss = 0
	epoch_acc = 0
	print(len(data_loader))
	for i, data in enumerate(data_loader, 0):
		# if(i%10 == 0):
		# 	print(i)
		imgs, labels = data
		imgs = imgs.float().cuda()
		labels= labels.float().cuda()
		y_pred = net.forward(imgs)

		criterion = nn.CrossEntropyLoss()
		# print(labels)
		loss = criterion(y_pred, torch.max(labels, 1)[1])
		y_pred = F.softmax(y_pred)
		# print('Net output', y_pred)
		output = torch.max(y_pred, 1)[1]
		# print('output', output)
		# print('Labels', labels)
		ground_truth = torch.max(labels, 1)[1]
		# print('ground_truth', ground_truth)
		correct = ((output == ground_truth).float()).sum()

		epoch_loss = epoch_loss + loss
		epoch_acc = epoch_acc + correct.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return epoch_loss *1.0/len(data_loader), epoch_acc*1.0/(len(data_loader) * batch_size)

def evaluate(net, data_loader, batch_size):
	# train(net, data_loader, optimizer)
	val_acc = 0
	for i, data in enumerate(data_loader, 0):
		imgs, labels = data
		imgs = imgs.float().cuda()
		labels = labels.float().cuda()
		y_pred = net.forward(imgs)
		# y_pred = F.softmax(y_pred)
		output = torch.max(y_pred, 1)[1]
		# print(output)
		ground_truth = torch.max(labels, 1)[1]
		# print(ground_truth)
		correct = ((output == ground_truth).float()).sum()
		# print(correct)

		val_acc = val_acc + correct.item()

	return val_acc * 1.0/(len(data_loader)*batch_size)


def train_iter(data_train,data_val, net, num_iterations, batch_size, learning_rate, print_every, save_every):
	print(batch_size)
	train_loader = DataLoader(data_train, batch_size = int(batch_size), shuffle = True)
	val_loader =DataLoader(data_val, batch_size = int(batch_size), shuffle = True)

	optimizer = optim.Adam(net.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-8)

	for epoch in range(num_iterations):
		net = net.train().cuda()
		epoch_loss, epoch_acc = train(net, train_loader, optimizer, batch_size)

		net = net.eval()

		val_acc = evaluate(net, val_loader, batch_size)

		if(epoch % print_every == 0):
			print('Epoch {}: Loss {}, Train accuracy {}, Validation accuracy {}'.format(epoch+1, epoch_loss, epoch_acc, val_acc))

		if(epoch % save_every == 0):
			torch.save( net.state_dict(), 'models/sign_classifier2.pt')

def evaluate_single(image, net):
	net = net.eval().cuda()
	image = cv2.resize(image, (32, 32))
	# cv2.imshow('a',image)
	# cv2.waitKey(0)
	image = np.resize(image, (1, image.shape[0], image.shape[1], image.shape[2]))
	# print(image.shape)
	image = np.array(image)
	image = image.transpose([0, 3, 1, 2])
	image = torch.from_numpy(image)
	image = image.float().cuda()
	output = net.forward(image)
	# output = F.softmax(output)
	# print(output)
	output = torch.max(output, 1)[1]
	return output.item()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('num_iterations', type=int)
	parser.add_argument('batch_size', type=int)
	parser.add_argument('learning_rate', type=float)
	parser.add_argument('print_every', type=int)
	parser.add_argument('save_every', type=int)
	parser.add_argument('num_classes', type=int)

	args = parser.parse_args()

	 

	net = Net(args.num_classes)
	# net = net.cuda()
	# print('Model constructed')
	transform = transforms.Compose ([
    transforms.ToPILImage(),
    transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)], 0.99),
	transforms.RandomResizedCrop(32),
	# transforms.RandomCrop(32),
    transforms.RandomAffine(15),
    transforms.RandomRotation(30, resample=False, expand=False, center=None),
    # transforms.RandomHorizontalFlip(p=0.5),

	])
	data_train = dataset('train/', 'annotation_train.csv', transform)
	data_val = dataset('val/', 'annotation_val.csv', transform)

	train_iter(data_train, data_val, net, args.num_iterations, args.batch_size, args.learning_rate, args.print_every, args.save_every)
	# net.load_state_dict(torch.load('models/sign_classifier.pt'))
	# net = net.eval().cuda()
	# img = cv2.imread('3.png')
	# print(evaluate_single(img, net))
if __name__=='__main__':
	main()