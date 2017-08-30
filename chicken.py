# Using Python 3.6.1

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import InputLayer, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
from keras import callbacks
from keras.utils import plot_model
import numpy as np
from scipy.misc import imsave

working_dir = '/home/max/Dokumente/Seminarprojekt'

#%% Load Data
training_data = ImageDataGenerator(rescale=1./255,horizontal_flip=True,width_shift_range=0.2,height_shift_range=0.2).flow_from_directory(
        working_dir+'/data/training',
        target_size=(240,320),
        batch_size=15
        )

test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
        working_dir+'/data/test',
        target_size=(240,320),
        batch_size=15)
#%% Model architecture


# Only two convolutional layers with max pooling in between. 
def create_model(width, height, channels):
    
    model = Sequential()
        
    model.add( InputLayer(input_shape=(height,width,channels), dtype='float32') )
    
    model.add( Convolution2D(15, kernel_size=5) )
    model.add( Activation('relu') )
    
    model.add(MaxPooling2D(pool_size=(2,2)))
        
    model.add( Convolution2D(20, kernel_size=5) )
    model.add( Activation('relu') )
        
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    
    return model

# Three convolutional layers with higher kernel count.
def create_model_2(width, height, channels):
    
    model = Sequential()
        
    model.add( InputLayer(input_shape=(height,width,channels), dtype='float32') )
    
    model.add( Convolution2D(20, kernel_size=5) )
    model.add( Activation('relu') )
    
    model.add(MaxPooling2D(pool_size=(2,2)))
        
    model.add( Convolution2D(30, kernel_size=5) )
    model.add( Activation('relu') )
        
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add( Convolution2D(50, kernel_size=3) )
    model.add( Activation('relu') )
        
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='softmax'))
    
    return model
#%%
# Grabs the output of the desired network layer, given a certain input.
def get_output(model, layer, input):
    
    output = K.function( [model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    return output( [input,0] )[0]

# Utility function for displaying the output (feature maps).
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((0, 1, 2))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#%% Creation of the actual Keras model, compilation.

model = create_model_2(320,240,3)
print(model.output_shape)
#model.load_weights(working_dir+'/weights/first_try.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%% Fitting of the model.

checkpoint = [callbacks.ModelCheckpoint(working_dir+'/weights/weights.h5', monitor='val_acc', verbose=1,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)]
model.fit_generator(
        training_data,
        validation_data=test_data,
        steps_per_epoch=22,
        validation_steps=3,
        epochs=10,
        callbacks=checkpoint)

#%% Score test for one test set batch.

testset = test_data.next()
score = model.evaluate(testset[0],testset[1])
print(score)

#%% Here the trained model is applied to an image.

img = load_img(working_dir+'/data/test/3/2017-06-11_18:55:05.jpg',target_size=(240,320))
img = img_to_array(img)
img = img.reshape((1,) + img.shape)
model.predict(img)

#%% This ouputs the feature maps for a certain input image and saves them on disk.

img = test_data.next()[0][0]
#img = load_img(working_dir+'/data/test/2/2017-06-11_18:22:45.jpg',target_size=(240,320))
#img = img_to_array(img)
imsave(working_dir+'/output/original.png', img)
img = img.reshape((1,) + img.shape)
print(model.predict(img))
for layer in [2,5]:
    out = get_output(model, layer, img)
    out = deprocess_image(out[0])
    for filter in range(out.shape[2]):
        imsave(working_dir+'/output/layer_%d_filter_%d.png' % (layer, filter), out[:,:,filter])
        
#%% 
model.save_weights(working_dir+'/weights/simple_model.h5')
#%%
plot_model(model, to_file=working_dir+'/model.svg', show_layer_names=False, show_shapes=False)
