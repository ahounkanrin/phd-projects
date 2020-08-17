import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py


DIR = "/scratch/hnkmah001/Datasets/ctfullbody/larger_fov_with_background/"

def save_dataset(x_train, y_train, x_val, y_val, x_test, y_test, fdir):
    hf = h5py.File(fdir, 'w')
    hf.create_dataset('x_train', data=x_train)
    hf.create_dataset('y_train', data=y_train)
    hf.create_dataset('x_val', data=x_val)
    hf.create_dataset('y_val', data=y_val)
    hf.create_dataset('x_test', data=x_test)
    hf.create_dataset('y_test', data=y_test)
    hf.close()

def soft_labels_encoder(angle, stdev=10):
	assert angle<360, "viewpoint angle must be less than 360"
	assert stdev%2 == 0, "stdev must be even"
	labels = np.zeros(360)
	central_prob = 0.2
	labels[angle] = central_prob
	for i in range(1, stdev//2):
		labels[(angle-i)%360] = (1-central_prob)/(stdev-2)
		labels[(angle+i)%360] = (1-central_prob)/(stdev-2)		
	return labels

def soft_labels_encoder_v2(angle):
	assert angle<360, "viewpoint angle must be less than 360"
	
	labels = np.zeros(360)
	labels[angle] = 0.3
	labels[(angle-1)%360] = 0.2
	labels[(angle-2)%360] = 0.1
	labels[(angle-3)%360] = 0.05
	labels[(angle+1)%360] = 0.2
	labels[(angle+2)%360] = 0.1
	labels[(angle+3)%360] = 0.05
				
	return labels



print('[INFO] Loading CSV files...')
train_csv = pd.read_csv('train_rz.csv', sep=',')
train_img_path = train_csv["impath"].tolist()
train_img_label = train_csv["label"].tolist()
val_csv = pd.read_csv('val_rz.csv', sep=',')
val_img_path = val_csv["impath"].tolist()
val_img_label = val_csv["label"].tolist()
test_csv = pd.read_csv('test_rz.csv', sep=',')
test_img_path = test_csv["impath"].tolist()
test_img_label = test_csv["label"].tolist()

print('[INFO] Preprocessing x_train...')
x_train = np.zeros([len(train_img_path), 400, 400, 3], dtype=np.uint8)
y_train = []

for i in tqdm(range(len(train_img_path))): # len(train_img_path))
    img = cv2.imread(DIR+train_img_path[i], 1)
    x_train[i] = img
    y_train.append(int(train_img_label[i])-1)
y_train = np.array(y_train)
#y_train = tf.keras.backend.one_hot(y_train, num_classes=360)


print('[INFO] Preprocessing x_val...')
x_val = np.zeros([len(val_img_path), 400, 400, 3], dtype=np.uint8)
y_val = []
for i in tqdm(range(len(val_img_path))):  # len(val_img_path)
    img = cv2.imread(DIR+val_img_path[i], 1)
    x_val[i] = img
    y_val.append(int(val_img_label[i])-1)
#y_val = tf.keras.backend.one_hot(y_val, num_classes=360)
y_val = np.array(y_val)

print('[INFO] Preprocessing x_test...')
x_test = np.zeros([len(test_img_path), 400, 400,3], dtype=np.uint8)
y_test = []
for i in tqdm(range(len(test_img_path))):
    img = cv2.imread(DIR+test_img_path[i], 1)
    x_test[i] = img
    y_test.append(int(test_img_label[i])-1)
#y_test = tf.keras.backend.one_hot(y_test, num_classes=360)
y_test = np.array(y_test)
save_dataset(x_train, y_train,x_val, y_val, x_test, y_test, DIR + 'chest_fov_400x400_onehot.h5')
