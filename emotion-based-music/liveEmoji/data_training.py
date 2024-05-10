# import os
# import numpy as np
# import cv
# from tensorflow.keras.utils import to_categorical
#
# from keras.layers import Input, Dense
# from keras.models import Model
#
# is_init = False
# size = -1
#
# label = []
# dictionary = {}
# c = 0
#
# for i in os.listdir():
# 	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
# 		if not(is_init):
# 			is_init = True
# 			X = np.load(i)
# 			size = X.shape[0]
# 			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
# 		else:
# 			X = np.concatenate((X, np.load(i)))
# 			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))
#
# 		label.append(i.split('.')[0])
# 		dictionary[i.split('.')[0]] = c
# 		c = c+1
#
#
# for i in range(y.shape[0]):
# 	y[i, 0] = dictionary[y[i, 0]]
# y = np.array(y, dtype="int32")
#
# ###  hello = 0 nope = 1 ---> [1,0] ... [0,1]
#
# y = to_categorical(y)
# X_new = X.copy()
# y_new = y.copy()
# counter = 0
#
# cnt = np.arange(X.shape[0])
# np.random.shuffle(cnt)
#
# for i in cnt:
# 	X_new[counter] = X[i]
# 	y_new[counter] = y[i]
# 	counter = counter + 1
#
#
# ip = Input(shape=(X.shape[1]))
#
# m = Dense(512, activation="relu")(ip)
# m = Dense(256, activation="relu")(m)
# op = Dense(y.shape[1], activation="softmax")(m)
# model = Model(inputs=ip, outputs=op)
# model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
# model.fit(X, y, epochs=50)
#
#
# model.save("model.h5")
# np.save("labels.npy", np.array(label))
#









import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables
label = []
dictionary = {}
c = 0

# Load data from .npy files
for file_name in os.listdir():
    if file_name.endswith(".npy") and not file_name.startswith("labels"):
        data = np.load(file_name)
        if not label:
            X = data
            size = data.shape[0]
        else:
            X = np.concatenate((X, data))
        y = np.full((data.shape[0], 1), c)
        label.append(file_name.split(".")[0])
        dictionary[file_name.split(".")[0]] = c
        c += 1

# Convert labels to one-hot encoding
y = to_categorical(y)

# Shuffle data
# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X = X[indices]
# y = y[indices]
# Shuffle data
num_samples = X.shape[0]
indices = np.random.choice(num_samples, num_samples, replace=False)
X = X[indices]
y = y[indices]

# Define model architecture
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)
model = Model(inputs=ip, outputs=op)

# Compile and train the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
model.fit(X, y, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))










# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from keras.layers import Input, Dense
# from keras.models import Model
#
# # Initialize variables
# label = []
# dictionary = {}
# c = 0
#
# # Load data from .npy files
# for file_name in os.listdir():
#     if file_name.endswith(".npy") and not file_name.startswith("labels"):
#         data = np.load(file_name)
#         if c == 0:
#             X = data
#             size = data.shape[0]
#         else:
#             X = np.concatenate((X, data))
#         y = np.array([c] * data.shape[0])
#         label.append(file_name.split(".")[0])
#         dictionary[file_name.split(".")[0]] = c
#         c += 1
#
# # Convert labels to one-hot encoding
# y = to_categorical(y)
#
# # Shuffle and split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define model architecture
# ip = Input(shape=(X_train.shape[1],))
# m = Dense(512, activation="relu")(ip)
# m = Dense(256, activation="relu")(m)
# op = Dense(y_train.shape[1], activation="softmax")(m)
# model = Model(inputs=ip, outputs=op)
#
# # Compile and train the model
# model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
# model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
#
# # Save the model and labels
# model.save("model.h5")
# np.save("labels.npy", np.array(label))
