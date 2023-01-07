from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matlab.engine
from sklearn.metrics import roc_auc_score,roc_curve, auc,confusion_matrix
import matplotlib.pyplot as plt
from numpy import *

eng = matlab.engine.start_matlab()
loss_object = tf.keras.losses.MeanSquaredError()
loss_object2 = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# gpu setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# encode network
def encode():
    inputs = tf.keras.layers.Input(shape=[50,] )   # input feature number = 50
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)


# decode network
def decode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(50, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(50,1))

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)

def adencode():
    inputs = tf.keras.layers.Input(shape=[20,] )   # input feature number = 50
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)


# decode network
def addecode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(20, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(20,1))

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)

def adtrain_step(images, labels):
    with tf.GradientTape() as adencode_tape, tf.GradientTape() as addecode_tape, tf.GradientTape() as classify_tape:
        y = adencoder(images)
        z = addecoder(y)
        predicted_label = classifier(y)

        loss1 = loss_object(images, z)
        loss2 = loss_object2(labels, predicted_label)
        loss_sum = loss1 + loss2

    gradient_e = adencode_tape.gradient(loss_sum, adencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, adencoder.trainable_variables))

    gradient_d = addecode_tape.gradient(loss1, addecoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, addecoder.trainable_variables))

    gradient_c = classify_tape.gradient(loss2, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return loss1, loss2, predicted_label
#subtype h0
def adtrain_h0(train_data, train_label,Batch_size ,print_information=False):
    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, 20))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, predicted_label = adtrain_step(images=train_x, labels=label_x)

        train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Accuracy : {}%, {}%, {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2)
            loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1, loss2, train_accuracy.result() * 100, loss1_account_for,
                                   loss2_account_for))

    y_h0_x = adencoder(train_data)
    tf.keras.backend.clear_session()
    return y_h0_x, train_label
#subtype h1
def adtrain_h1(train_data, train_label,Batch_size, print_information=False):

    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, 20))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, predicted_label = adtrain_step(images=train_x, labels=label_x)

        train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse {}, Loss2 cross entropy: {}, Accuracy : {}%, {}%, {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2)
            loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1, loss2, train_accuracy.result() * 100, loss1_account_for,
                                   loss2_account_for))

    y_h1_x = adencoder(train_data)
    tf.keras.backend.clear_session()
    return y_h1_x, train_label

# residual_block for classification of hidden feature
def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()  # 采用sequential构造法
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result


# classification network
def classify():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])

    block_stack_1 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_2 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_3 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]

    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer_in = tf.keras.layers.Dense(num_of_hidden_classify, kernel_initializer='he_normal', activation='relu')
    layer_out = tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')

    res_x_0 = 0
    res_x_1 = 0
    res_x_2 = 0

    x = inputs
    x = layer0(x)
    x = layer_in(x)

    x_0 = x
    for block in block_stack_1:
        res_x_0 = block(x)
    x = res_x_0 + x

    for block in block_stack_2:
        res_x_1 = block(x)
    x = res_x_1 + x

    for block in block_stack_3:
        res_x_2 = block(x)
    x = res_x_2 + x

    x = x_0 + x
    x = layer_out(x)   # output dimension: 2
    return tf.keras.Model(inputs=inputs, outputs=x)


def train_step(images, labels):
    with tf.GradientTape() as encode_tape, tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape:
        y = encoder(images)
        z = decoder(y)
        predicted_label = classifier(y)

        loss1 = loss_object(images, z)
        loss2 = loss_object2(labels, predicted_label)
        loss_sum = loss1 + loss2

    gradient_e = encode_tape.gradient(loss_sum, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    gradient_d = decode_tape.gradient(loss1, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))

    gradient_c = classify_tape.gradient(loss2, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return loss1, loss2, predicted_label


def prepare_data(index, data_name):

    # get functional connections and their labels by matlab code
    train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label,testlabel= eng.svm_two_suppose_FC(index, data_name,nargout=7)

    train_h0_data = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h1_data = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)
    test_h0_label = np.array(test_h0_label)
    test_h1_label = np.array(test_h1_label)
    testlabel = np.array(testlabel)

    num_h0 = train_h0_label.sum()  # ADHD subjects in h0 hypothesis
    num_h1 = train_h1_label.sum()  # ADHD subjects in h1 hypothesis

    return train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label, testlabel

def prepare_adhddata(index, data_name):

    # get functional connections and their labels by matlab code
    train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label = eng.svm_adad(index, data_name, nargout=6)

    train_h0_data = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h1_data = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)
    test_h0_label = np.array(test_h0_label)
    test_h1_label = np.array(test_h1_label)

    num_h0 = train_h0_label.sum()  # ADHD subjects in h0 hypothesis
    num_h1 = train_h1_label.sum()  # ADHD subjects in h1 hypothesis

    return train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label

def prepare_hcaddata(index, data_name):

    # get functional connections and their labels by matlab code
    train_h0_data, train_h0_label, train_h1_data, train_h1_label= eng.svm_hcad(index, data_name, nargout=4)

    train_h0_data = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h1_data = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)

    num_h0 = train_h0_label.sum()  # ADHD subjects in h0 hypothesis
    num_h1 = train_h1_label.sum()  # ADHD subjects in h1 hypothesis

    return train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1



# h0 training model
def train_h0(train_data, train_label,Batch_size ,print_information=False):
    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, 50))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, predicted_label = train_step(images=train_x, labels=label_x)

        train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Accuracy : {}%, {}%, {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2)
            loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1, loss2, train_accuracy.result() * 100, loss1_account_for,
                                   loss2_account_for))

    y_h0_x = encoder(train_data)
    tf.keras.backend.clear_session()
    return y_h0_x, train_label


# h1 training model
def train_h1(train_data, train_label,Batch_size, print_information=False):

    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, 50))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, predicted_label = train_step(images=train_x, labels=label_x)

        train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse {}, Loss2 cross entropy: {}, Accuracy : {}%, {}%, {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2)
            loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1, loss2, train_accuracy.result() * 100, loss1_account_for,
                                   loss2_account_for))

    y_h1_x = encoder(train_data)
    tf.keras.backend.clear_session()
    return y_h1_x, train_label


def judge2(y_h0_x, h0_label, y_h1_x, h1_label, num_h0, num_h1):
    if h0_label is None:
        if h1_label is None:
            pass

    # h0
    yh0_np = np.array(y_h0_x)  # deeper feature in h0
    yh0_AD = np.split(yh0_np, (num_h0,))
    yh0_AD = np.array(yh0_AD)
    yh0_HC = np.copy(yh0_AD)
    yh0_AD = np.delete(yh0_AD, 1, axis=0)[0]
    yh0_HC = np.delete(yh0_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh0_AD_avg = np.mean(yh0_AD, axis=(0,))
    yh0_HC_avg = np.mean(yh0_HC, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))

    yh0_intra_AD = np.sum(np.power(np.linalg.norm((yh0_AD - yh0_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_HC = np.sum(np.power(np.linalg.norm((yh0_HC - yh0_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_AD + yh0_intra_HC

    yh0_inter_AD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AD_avg), axis=0, keepdims=True), 2))
    yh0_inter_HC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_HC_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = num_h0 * yh0_inter_AD + (yh0_np.shape[0] - num_h0) * yh0_inter_HC

    yh0_out_class = yh0_intra_all / yh0_inter_all

    # h1
    yh1_np = np.array(y_h1_x)  # deeper feature in h1
    yh1_AD = np.split(yh1_np, (num_h1,))
    yh1_AD = np.array(yh1_AD)
    yh1_HC = np.copy(yh1_AD)
    yh1_AD = np.delete(yh1_AD, 1, axis=0)[0]
    yh1_HC = np.delete(yh1_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh1_AD_avg = np.mean(yh1_AD, axis=(0,))  # h1 ADHD均值
    yh1_HC_avg = np.mean(yh1_HC, axis=(0,))  # h1 HC均值
    yh1_all_avg = np.mean(yh1_np, axis=(0,))  # 总均值

    yh1_intra_AD = np.sum(np.power(np.linalg.norm((yh1_AD - yh1_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_HC = np.sum(np.power(np.linalg.norm((yh1_HC - yh1_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_all = yh1_intra_AD + yh1_intra_HC

    yh1_inter_AD = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_AD_avg), axis=0, keepdims=True), 2))
    yh1_inter_HC = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_HC_avg), axis=0, keepdims=True), 2))
    yh1_inter_all = num_h1 * yh1_inter_AD + (yh1_np.shape[0] - num_h1) * yh1_inter_HC

    yh1_out_class = yh1_intra_all / yh1_inter_all

    # ADHD decision function
    if yh1_out_class >= yh0_out_class:
        return True
    else:
        return False



def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num / den


if __name__ == '__main__':
    name_list = ['NYU_data', 'Peking_data', 'KKI_data', 'NI_data', 'Peking_1_data']
    dict_data = {'NYU_data': 216, 'Peking_data': 194, 'KKI_data': 83, 'NI_data': 48, 'Peking_1_data': 86}
    EPOCH_list = {'NYU_data': 100, 'Peking_data': 100, 'KKI_data': 50, 'NI_data': 35, 'Peking_1_data': 50}
    addict_data = {'NYU_data': 118, 'Peking_data': 78, 'KKI_data': 22, 'NI_data': 25, 'Peking_1_data': 24}
    for i_out in range(0, 1):   # select ADHD-200 datasets
        time=30
        name_of_data = name_list[i_out]
        num_of_hidden = 30              # neural unit in auto-coding network
        num_of_hidden_classify = 20     # neural unit in classification network
        Batch_size = dict_data[name_of_data] - 1
        adBatch_size = addict_data[name_of_data] - 1
        hcadBatch_size = addict_data[name_of_data]
        EPOCH = EPOCH_list[name_of_data]
        accuracy=[]

        for j_out in range(time):
            encoder=encode()
            decoder=decode()
            adencoder=adencode()
            addecoder=addecode()
            classifier = classify()
            m = 0
            j = 0
            k = 0

            hchc=0
            ad1ad1 =0
            ad1ad3 = 0
            ad3ad3 = 0
            ad3ad1 = 0
            ad1hc = 0
            ad3hc = 0
            hcad1=0
            hcad3=0

            for i in range(dict_data[name_of_data]):

                train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label ,testlabel= prepare_data(
                    index=i + 1, data_name=name_of_data)


                y_h0, train_label_h0 = train_h0(train_h0_data, train_h0_label, Batch_size, print_information=False)
                tf.keras.backend.clear_session()
                y_h1, train_label_h1 = train_h1(train_h1_data, train_h1_label, Batch_size, print_information=False)
                tf.keras.backend.clear_session()

                judge_result2 = judge2(y_h0, train_h0_label, y_h1, train_h1_label, num_h0, num_h1)

                if judge_result2:
                    k += 1
                if judge_result2 == True and test_h0_label == 2:
                    j += 1
                    hchc=hchc+1
                #if true label is ADHD,predicted label is ADHD
                if judge_result2 == True and test_h0_label == 1:
                    adtrain_h0_data, adtrain_h0_label, adtrain_h1_data, adtrain_h1_label, adnum_h0, adnum_h1, adtest_h0_label, adtest_h1_label = prepare_adhddata(
                        index=i + 1, data_name=name_of_data)
                    ady_h0, adtrain_label_h0 = adtrain_h0(adtrain_h0_data, adtrain_h0_label, adBatch_size,
                                                          print_information=False)
                    tf.keras.backend.clear_session()
                    ady_h1, adtrain_label_h1 = adtrain_h1(adtrain_h1_data, adtrain_h1_label, adBatch_size,
                                                          print_information=False)
                    tf.keras.backend.clear_session()
                    adjudge_result2 = judge2(ady_h0, adtrain_h0_label, ady_h1, adtrain_h1_label, adnum_h0, adnum_h1)
                    if adjudge_result2:
                        m += 1
                    if adjudge_result2 == True and adtest_h0_label == 2:

                        ad1ad1+=1

                    if adjudge_result2 == True and adtest_h0_label == 1:

                        ad3ad3+=1

                    if adjudge_result2 == False and testlabel == 2:

                        ad1ad3+=1

                    if adjudge_result2 == False and testlabel== 1:

                        ad3ad1+=1

                    j += 1

                # if true label is HC,predicted label is ADHD
                if judge_result2 == False and test_h0_label == 2:


                    hcadtrain_h0_data, hcadtrain_h0_label, hcadtrain_h1_data, hcadtrain_h1_label, hcadnum_h0, hcadnum_h1 = prepare_hcaddata(
                        index=i + 1, data_name=name_of_data)
                    hcady_h0, hcadtrain_label_h0 = adtrain_h0(hcadtrain_h0_data, hcadtrain_h0_label, hcadBatch_size,
                                                              print_information=False)
                    tf.keras.backend.clear_session()
                    hcady_h1, hcadtrain_label_h1 = adtrain_h1(hcadtrain_h1_data, hcadtrain_h1_label, hcadBatch_size,
                                                              print_information=False)
                    tf.keras.backend.clear_session()
                    hcadjudge_result2 = judge2(hcady_h0, hcadtrain_h0_label, hcady_h1, hcadtrain_h1_label,
                                               hcadnum_h0,
                                               hcadnum_h1)
                    if hcadjudge_result2:
                        hcad1 += 1

                    if hcadjudge_result2 == False:
                        hcad3 += 1


                if judge_result2 == False and test_h0_label == 1:

                    if testlabel == 2:
                        ad1hc += 1

                    if testlabel == 1:
                        ad3hc += 1





                print('\n current loop:' + str(i + 1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
                print('-------------' + str(j_out + 1) + ' / ' + '50' + '-------------\n')

            accuracy.append((hchc+m) / dict_data[name_of_data] )
            results_txt = str(hchc) + '\t' + str(ad1ad1) + '\t' + str(ad3ad3) + '\t' + str(
                ad1ad3) + '\t' + str(ad3ad1) + '\t' + str(ad3hc) + '\t' +  str(ad1hc) + '\t'+str(hcad1) + '\t' + str(hcad3) + '\t' + str(
                100 * (hchc+m) / dict_data[name_of_data]) + '\t' +'\n'

            with open('./result/' + name_of_data + '.txt', "a+") as f:
                f.write(results_txt)

            del encoder
            del decoder
            del adencoder
            del addecoder
            del classifier
