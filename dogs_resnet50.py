import numpy as np 
import dog_loader
import torchvision
import torch
from torchvision import transforms, utils
from torchvision.models import resnet
# ====================== GLOBAL VARIABLES =====================================
input_shape = (224, 224, 3)
learning_rate = 0.0007
batch_size = 32
num_epochs = 100
all_transforms = transforms.Compose([transforms.Resize(input_shape[:2]), transforms.RandomHorizontalFlip(0.5),transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
	transforms.RandomResizedCrop(input_shape[:2]), transforms.ToTensor()])

use_cuda = torch.cuda.is_available()
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 4, "pin_memory": True}
    print("CUDA is supported")
else:
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

train_loader, val_loader, test_loader = dog_loader.create_loaders(batch_size, 
										transform=all_transforms, shuffle=True, 
										extras=extras)
	
# ======================= THE MODEL ==============================================
model = resnet.resnet50(num_classes=120)
model.to(computing_device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
	model.train()
	train_loss, valid_loss = [], []
	train_acc, valid_acc = [], []
	for batch_id, (X_batch, y_batch) in enumerate(train_loader, 0):
		X_batch, y_batch = X_batch.to(computing_device), y_batch.to(computing_device)
		optimizer.zero_grad()

		outputs = model(X_batch)
		loss = criterion(outputs, y_batch)
		_, predicted = torch.max(outputs.data, 1)
		acc = (predicted == y_batch).sum().item()*1./batch_size

		loss.backward()
		optimizer.step()

		train_loss.append(loss.item())
 		train_acc.append(acc)
 		print 'Epoch {} Iteration {} Loss {} Accuracy {}'.format(epoch, batch_id, loss.item(), acc)

	print 'Validating now...'
	model.eval()
	for batch_id, (X_batch, y_batch) in enumerate(val_loader, 0):
		X_batch, y_batch = X_batch.to(computing_device), y_batch.to(computing_device)
		outputs = model(X_batch)
		loss = criterion(outputs, y_batch).item()
		_, predicted = torch.max(outputs.data, 1)
		acc = (predicted == y_batch).sum().item()*1./batch_size
		
		valid_loss.append(loss)
		valid_acc.append(acc)
		print 'Validation Epoch {} Iteration {} Validation Loss {} Validation Accuracy {}'.format(epoch, batch_id, loss, acc)

	print 'Avg Training Loss {} Accuracy {} Test Loss {} Accuracy {}'.format(np.mean(train_loss), np.mean(train_acc), np.mean(valid_loss), np.mean(valid_acc))
	torch.save(model.state_dict(), './models/epoch{}_validloss{}_validacc{}'.format(epoch, np.mean(valid_loss), np.mean(valid_acc)))

	# print 'Visualizing now...'
	# saver.restore(sess, './models/birds_epoch19_validloss1.04835760593_validacc0.744140625')
	# # samples = random.sample(range(X_train.shape[0]), 10) # take 10 random samples from the training_set
	# samples = range(100)
	# for idx in samples:
	# 	this_sample = X_valid[idx].reshape((1,) + X_valid[idx].shape)
	# 	draw_CAM(idx, this_sample, sess)
