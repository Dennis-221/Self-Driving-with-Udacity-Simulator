import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utilis import *
from sklearn.model_selection import train_test_split

print('SETTING UP ..........')
########STEP 1
path = 'My Training Data'
data = importDataInfo(path)

########STEP 2
balanceData(data, display=True)

########STEP 3
imagesPath, steerings = loadData(path, data)

# print(imagesPath[0])
# print(steerings[0])
#
#########STEP 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total Training images:', len(xTrain))
print('Total Validation images:', len(xVal))


#########STEP 5

#########STEP 6 : Preprocessing

#########STEP 7 : Batch Generation

#########STEP 8 : Model Creation
model = createModel()
model.summary()

#########STEP 9 :
history = model.fit(batchGen(xTrain,yTrain,100,1), steps_per_epoch=300, epochs=20,
          validation_data=batchGen(xVal,yVal,100,0), validation_steps=200)

#########STEP 10 : SAVE THE MODEL
model.save('model_l1.h5')
print('Model Saved.')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0,0.4])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
