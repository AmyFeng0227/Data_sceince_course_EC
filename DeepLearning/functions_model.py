
# TensorFlow and Keras imports
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam


def build_model_1(image_size, learning_rate, color_input):
    model = Sequential()

    #1st CNN layer
    model.add(Conv2D(64,(3,3), padding="same",input_shape=(image_size,image_size,color_input)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #2nd CNN layer
    model.add(Conv2D(128,(5,5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #3rd CNN layer
    model.add(Conv2D(512,(3,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #4th CNN layer
    model.add(Conv2D(512,(3,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #5th CNN layer
    model.add(Conv2D(512,(3,3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #flatten layer
    model.add(Flatten())

    #fully connected 1st layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))


    #fully connected 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #fully connected 3rd layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    #output layer
    model.add(Dense(7, activation='softmax'))

    print("Model is created.")
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Model is compiled.\nModel completed.")
    return model

#class WandbCallback(Callback):
 #   def on_epoch_end(self, epoch, logs=None):
     #   wandb.log(logs)


def create_callbacks_list(folder_path,model_name):

    checkpoint = ModelCheckpoint(folder_path + "/"+ model_name + ".h5",
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=5,
                                   verbose=1,
                                   restore_best_weights=True
                                )

    reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.2,
                                           patience=3,
                                           verbose=1,
                                           min_delta=0.0001
                                        )
   # wandb_callback = WandbCallback()
    callbacks_list = [early_stopping, checkpoint, reduce_learningrate]
    print("Callbacks_list created.")
    print("checkpoint name:" + model_name + ".h5")
    return callbacks_list

def fit_model(model, training_dataset, validation_dataset, epochs, callbacks_list, batch_size, model_name, folder_path):
    print("Dataset size:", len(training_dataset))
        # Initialize WandB run
    #wandb.init(
        #project="kunskapskontroll_deep_learning",
        #config={
        #    "learning_rate": model.optimizer.lr.numpy(),
       #     "architecture": "CNN",
       #     "dataset": dataset_name,  # Update with your dataset name
        #    "epochs": epochs
       # }
   # )
    training_history = model.fit(training_dataset,
                                      steps_per_epoch = training_dataset.samples // batch_size,
                                      epochs=epochs,
                                       validation_data=validation_dataset,
                                       validation_steps = validation_dataset.samples // batch_size,

                                       callbacks=callbacks_list
                                       )
    print("Training completed.")
    model.save(folder_path+"/" + model_name +".h5")
    print("Model training completed and saved.")
    print(training_history)
    #wandb.finish()
    return training_history



