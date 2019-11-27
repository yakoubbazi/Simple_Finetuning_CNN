######################################################################
# Scene classification: PhD Course:   ------- 2nd Sems. January 2019
#########################################################################

# Here we are using parts only of the squizzenet

from keras.layers.merge import concatenate
import numpy as np
import scipy.io     # read mat flies
from keras import backend as K
K.set_image_data_format('channels_last')
import gc
from tensorflow import set_random_seed
set_random_seed(1)
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, BatchNormalization, Add, GlobalAveragePooling1D, MaxPooling2D
from keras.layers import LeakyReLU, Conv1D,GlobalMaxPooling2D, GlobalAveragePooling2D,Lambda, Multiply, Bidirectional, merge, Concatenate
from keras.layers import Permute, GlobalMaxPooling1D, AtrousConvolution2D, Conv2D, Conv2DTranspose, UpSampling2D, GRU,Bidirectional, Dot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.applications import nasnet, xception, inception_resnet_v2, densenet, vgg16,vgg19
from keras.applications import inception_v3,inception_resnet_v2
from keras.models import Model
from Read_batch_data import Split_train_test_yakoub
from keras.utils import  to_categorical
import time
import copy
from keras_efficientnets import efficientnet
from keras_efficientnets.custom_objects import Swish
from keras import layers
import inception_v1
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot



def Top_ResNet2(num_classes,model_type):
    # feature extractor (encoder)
    if model_type==1:
        inputs1 = Input(shape=(16,16,832))
        inputs2 = Input(shape=(8,8,1024))
    if model_type==2:
        inputs1 = Input(shape=(14,14,768))
        inputs2 = Input(shape=(6,6,2048))
    if model_type==3:
        inputs1 = Input(shape=(14,14,1088))
        inputs2 = Input(shape=(6,6,1536))
    if model_type == 4:
        inputs1 = Input(shape=(16, 16, 672))
        inputs2 = Input(shape=(8, 8, 1280))
        dimT=672
    if model_type == 5:
        inputs1 = Input(shape=(16, 16, 1056))
        inputs2 = Input(shape=(8, 8, 2048))
    if model_type == 6:
        inputs1 = Input(shape=(16, 16, 816))
        inputs2 = Input(shape=(8, 8, 1536))
    if model_type == 7:
        inputs1 = Input(shape=(16, 16, 640))
        inputs2 = Input(shape=(8, 8, 1664))

    if model_type == 8:
        inputs1 = Input(shape=(16, 16, 640))
        inputs2 = Input(shape=(8, 8, 1792))

    xx1 = Conv2D(128, 3, strides=2)(inputs1)
    xx1=BatchNormalization()(xx1)
    #xx1 = Activation('relu')(xx1)

    xx1=Swish()(xx1)
    xx1 = GlobalAveragePooling2D(name='fea_out1')(xx1)
    xx1 = Dropout(0.8)(xx1)
    output1 = Dense(num_classes, activation='linear')(xx1)
    output1=Activation('softmax')(output1)

    xx2 = GlobalAveragePooling2D(name='fea_out2')(inputs2)
    xx2 = Dense(num_classes, activation='linear')(xx2)
    output2 = Activation('softmax')(xx2)

    model = Model(inputs=[inputs1,inputs2], outputs=[output1,output2])
    print(model.summary())

    return model



# Define main code here
if __name__ == '__main__':

    print('my code start here.................')
    time_start = time.clock()

    img_rows, img_cols = 256, 256
    img_dim = (img_rows, img_rows, 3)
    epochs = 40
    Percent_train=80                   # Training percentage
    model_type=6
    numiter=5
    batch_size=25      #### for some models we use 50 due to memory limit
    Use_data_aug=0
    lam_loss=[0.5,0.5]

    # Split data randomly into training and Testing index
    #Tridx = []
    #Tsidx = []
    #numiter=5
    #for jk in range(numiter):
    #    [Train_index, Test_index] = Split_train_test_yakoub(Num_class=num_classes, Ytask=y_task,
    #                                    Percent_train=Percent_train)
    #    Tridx.append(Train_index)
    #    Tsidx.append(Test_index)

    ###### Use Saved data split
    #np.savez('data/WHU_Train40_Test.npz', Tridx=Tridx, Tsidx=Tsidx)
    Tridx = []
    Tsidx = []
    #Data_idx=np.load('data/KSA_Train20_Test.npz')
    Data_idx=np.load('data/Optimal31_Train80_Test.npz')
    #Data_idx=np.load('data/AID_Train50_Test.npz')
    #Data_idx=np.load('data/Merced_Train50_Test.npz')
    #Data_idx=np.load('data/WHU_Train40_Test.npz')
    Tridx=Data_idx['Tridx']
    Tsidx = Data_idx['Tsidx']
    ##3 Dataset iteration.....................................
    Average1=[]
    Average2=[]
    Average3=[]
    Loss_5trials=[]
    for iter in range(numiter):
        # Get Train and Test data
        Xtrain_task1=[]
        Xtest_task1=[]
        mat = scipy.io.loadmat('data/Optimal31.mat')
        #mat = np.load('data/AID_256.npz')
       # mat = scipy.io.loadmat('data/KSA_256.mat')
        #mat = scipy.io.loadmat('data/Merced_256.mat')
        y_task=mat['y'][0]
        num_classes = y_task.max()
        y_task = y_task - 1
        y_task = to_categorical(y_task, num_classes)

        Xtrain_task1=mat['X'][Tridx[iter],:,:,:]
        Xtest_task1 = mat['X'][Tsidx[iter], :, :, :]
        ytrain_task1 = copy.deepcopy(y_task[Tridx[iter], :])
        ytest_task1 = copy.deepcopy(y_task[Tsidx[iter], :])

        if model_type==1:
            layer_name1='Mixed_4f_Concatenated'
            layer_name2='Mixed_5c_Concatenated'
            Xtest_task1 = inception_v1.preprocess_input(Xtest_task1.astype(np.float32))
            DNet=inception_v1.InceptionV1(include_top=False, weights='imagenet',input_shape=(img_dim[0], img_dim[0], 3))

        if model_type==2:
            layer_name1='mixed7'
            layer_name2='mixed10'
            Xtest_task1 = inception_v3.preprocess_input(Xtest_task1.astype(np.float32))
            DNet=inception_v3.InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=(img_dim[0], img_dim[0], 3),pooling=None,classes=1000)

        if model_type==3:
            layer_name1='block17_20_ac'
            layer_name2='conv_7b_ac'
            Xtest_task1 = inception_resnet_v2.preprocess_input(Xtest_task1.astype(np.float32))
            DNet = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                      input_shape=(img_dim[0], img_dim[1], 3), pooling=None, classes=1000)

        if model_type == 4:
            layer_name1 = 'swish_34'
            layer_name2 = 'swish_49'
            DNet = efficientnet.EfficientNetB0(img_dim, classes=1000, include_top=False, weights='imagenet')
            Xtest_task1 = efficientnet.preprocess_input(Xtest_task1.astype(np.float32))

        if model_type == 5:
            layer_name1 = 'swish_80'
            layer_name2 = 'swish_116'
            DNet = efficientnet.EfficientNetB5(img_dim, classes=1000, include_top=False, weights='imagenet')
            Xtest_task1 = efficientnet.preprocess_input(Xtest_task1.astype(np.float32))

        if model_type == 6:
            layer_name1 = 'swish_54'
            layer_name2 = 'swish_78'
            DNet = efficientnet.EfficientNetB3(img_dim, classes=1000, include_top=False, weights='imagenet')
            Xtest_task1 = efficientnet.preprocess_input(Xtest_task1.astype(np.float32))

        if model_type == 7:
            layer_name1 = 'pool4_conv'
            layer_name2 = 'relu'
            DNet = densenet.DenseNet169(include_top=False, weights='imagenet', input_tensor=None,
                                        input_shape=(256, 256, 3),
                                        pooling=None, classes=1000)
            Xtest_task1 = densenet.preprocess_input(Xtest_task1.astype(np.float32))



        ### Build model
        Top_resnet=Top_ResNet2(num_classes,model_type)
        DNet_new = Model(inputs=DNet.input, outputs=[DNet.get_layer(layer_name1).output,DNet.get_layer(layer_name2).output])
        Images1 = Input(shape=(img_dim[0], img_dim[0], 3))
        DNet_new_Out = DNet_new(Images1)
        Out_Top_ResNet=Top_resnet([DNet_new_Out[0],DNet_new_Out[1]])

        ######### Model with auxiliary classification loss........
        New_Res_model=Model(inputs=[Images1],outputs=[Out_Top_ResNet[0],Out_Top_ResNet[1]])
        New_Res_model.compile(loss=['categorical_crossentropy','categorical_crossentropy'],
                              loss_weights=[lam_loss[0],lam_loss[1]],optimizer=RMSprop(lr=0.0001))

        #datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
        #                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        #                             horizontal_flip=True, fill_mode="nearest")

        datagen = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True, fill_mode="nearest")
        ### Training procedure
        Total_loss=[]
        for epoch in range(epochs):

            print('Epoch {} of {}'.format(epoch + 1, epochs))
            nb_batches = int(Xtrain_task1.shape[0] / batch_size)

            idx_train_shuffle = np.arange(Xtrain_task1.shape[0])
            np.random.shuffle(idx_train_shuffle)

            if epoch==20:
                New_Res_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                                      loss_weights=[lam_loss[0],lam_loss[1]], optimizer=RMSprop(lr=0.00001))

            if epoch == 40:
                New_Res_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'],
                                      loss_weights=[lam_loss[0],lam_loss[1]], optimizer=RMSprop(lr=0.000001))

            epoch_loss = []

            index = 0

            while index < nb_batches:
                # idx = np.random.randint(0, Xtrain_task1.shape[0], 100)
                t1_data = []

                t1_data = copy.deepcopy(Xtrain_task1[idx_train_shuffle[index * batch_size:(index + 1) * batch_size]])
                t1_label = ytrain_task1[idx_train_shuffle[index * batch_size:(index + 1) * batch_size]]

                # If data augmentation is used
                if Use_data_aug==1:
                    t1_data_batch=[]
                    t1_label_batch=[]
                    for jtk in range(t1_data.shape[0]):
                        samples = np.expand_dims(t1_data[jtk], 0)
                        # prepare iterator
                        it = datagen.flow(samples, batch_size=1)
                        t1_data_batch.append(samples)
                        t1_label_batch.append(t1_label[jtk,:])
                        t1_data_batch.append(it.next())
                        t1_label_batch.append(t1_label[jtk, :])

                    t1_data_batch=np.array(t1_data_batch)
                    #print(t1_data_batch.shape)
                    t1_label_batch=np.array(t1_label_batch)
                    t1_data_batch=np.squeeze(t1_data_batch,axis=1)

                    t1_data_batch = efficientnet.preprocess_input(t1_data_batch.astype(np.float32))
                    Loss2 = New_Res_model.train_on_batch(t1_data_batch, [t1_label_batch,t1_label_batch])

                ### If no data_aug
                else:
                    t1_data = efficientnet.preprocess_input(t1_data.astype(np.float32))
                    Loss2 = New_Res_model.train_on_batch(t1_data, [t1_label, t1_label])

                epoch_loss.append(Loss2[0])

                index += 1


            Total_loss.append(np.mean(epoch_loss))
            if epoch>20:
                if Total_loss[epoch]<=0.002:
                    if np.sum(Total_loss[epoch-3:epoch])<=0.006:
                        break


            print('\n[Loss_1   : {:.3f}]'.format(np.mean(epoch_loss)))
        
        Loss_5trials.append(Total_loss)
        #Prob = Maxmaging_task.predict(Feat)
        print('predict on training data...............')
        Xtrain_task1=efficientnet.preprocess_input(Xtrain_task1.astype(np.float32))
        Prob = New_Res_model.predict(Xtrain_task1)
        Prob_label=np.argmax(Prob[1],axis=1)
        ytrain_true=np.argmax(ytrain_task1,axis=1)
        conf_matrix=confusion_matrix(ytrain_true, Prob_label)
        OA=100*np.sum(np.diag(conf_matrix))/np.sum(conf_matrix)
        print(OA)

        # Prob = Maxmaging_task.predict(Feat)
        print('predict on testing data ...............')
        Prob = New_Res_model.predict(Xtest_task1)
        Prob_label2 = np.argmax(Prob[1], axis=1)
        ytest_true = np.argmax(ytest_task1, axis=1)
        conf_matrix = confusion_matrix(ytest_true, Prob_label2)
        OA = 100 * np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        print(OA)
        Average2.append(OA)
        K.clear_session()
        gc.collect()
        del DNet, DNet_new, New_Res_model
        #DNet=[]
        #DNet_new=[]
        #New_Res_model=[]

    Total_OA2=np.mean(Average2)
    Total_std2 = np.std(Average2)

    print('predict overal testing data Five Trials...............')
    print(Total_OA2)
    print(Total_std2)

    time_elapsed = (time.clock() - time_start)/60
    print('Elapsed Time=',time_elapsed)

