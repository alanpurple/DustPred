import tensorflow as tf
from tensorflow.keras import layers,Model,Input,optimizers,metrics,Sequential,activations,models,losses,callbacks
import numpy as np
import h5py

from nsml import DATASET_PATH
import nsml

#import glob
import os
import argparse

FEATURE_DIM = 14 #지역(0~9), 연(2016~2019), 월, 일, t-5 ~ t-1의 미세 & 초미세
OUTPUT_DIM = 2 # t-time의 (미세, 초미세)

def bind_model(model):
    def save(path,**kwargs):
        # model.save(os.path.join(path,'alan_dust_model2.h5'))
        model.save_weights(os.path.join(path, 'model_alan.tf'),save_format='tf')

    def load(path):
        # model=models.load_model(os.path.join(path,'alan_dust_model2.h5'))
        model.load_weights(os.path.join(path,'model_alan.tf'))

    def infer(path):
        return inference(path,model)

    nsml.bind(save,load,infer)

def inference(path,model):
    test_path = path+'/test_data'
    data=np.load(test_path)
    #region=data[:,0]
    #month=data[:,2]
    test_dust=[[[elem[2*i],elem[2*i+1]] for i in range(2,7)] for elem in data]
    test_dust=np.asarray(test_dust)
    pred=model.predict(test_dust).tolist()
    result=[[i,elem[-1]] for i,elem in enumerate(pred)]
    return result

#def region_one_hot(region):
#    return tf.one_hot(region,10)

#def month_one_hot(month):
#    return tf.one_hot(month,12)

def get_simplegru_model(num_units=64,num_layers=2,dropout=0.1):
    input=Input((5,2),name='dust_input')
    inputs_expanded=layers.ZeroPadding1D((0,1))(input)
    #region=Input((),name='region_input',dtype=tf.uint8)
    #month=Input((),name='month_input',dtype=tf.float32)
    #cells=[layers.GRUCell(num_units) for _ in range(num_layers-1)]
    #cells.append(layers.GRUCell(num_units,dropout=dropout))
    #multi_gru=layers.RNN(cells,name='multi-lstm')
    multi_gru=layers.LSTM(num_units,activation='relu')

    # batch X num_units
    output=multi_gru(inputs_expanded)

    #  batch X 6 X num_units 
    repeated=layers.RepeatVector(6)(output)

    # batch X 6 X num_units
    lstm_seq=layers.LSTM(num_units,return_sequences=True,activation='relu')(repeated)

    #flattened=layers.Flatten()(inputs)

    # (batch,num_units)
    #feature=layers.Dense(num_units,activations.relu)(output)
    #region_one=layers.Lambda(region_one_hot,name='region_one_hot')(region)
    #month_one=layers.Lambda(month_one_hot,name='month_one_hot')(month)
    # (batch,num_units+10+12)
    #feature_concat=layers.Concatenate()([feature,month_one])
    #feature_concat=layers.Dropout(0.1)(feature_concat)
    #dense_final=layers.Dense(16,activations.relu)(feature)

    #tiny=layers.Dense(1,name='tiny_dense')(dense_final)
    #micro=layers.Dense(1,name='micro_dense')(dense_final)

    #model=Model(inputs=[inputs,region,month],outputs=[tiny,micro])

    final=layers.TimeDistributed(layers.Dense(2))(lstm_seq)
    model=Model(inputs=input,outputs=final)

    return model

EPOCHS=100

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    
    config = args.parse_args()

    model=get_simplegru_model()
    
    bind_model(model)

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())
    
    if config.mode == "train":

        model.compile(optimizer=optimizers.Adam(),
                    loss=losses.MeanSquaredError(),
                    metrics=[metrics.MeanSquaredError()])

        cbs=[
             callbacks.EarlyStopping(
                     monitor='val_loss',
                     min_delta=1e-2,patience=4,
                     restore_best_weights=True,
                     verbose=1)
             ]

        train_dataset_path = DATASET_PATH + '/train/train_data'

        train_label_file = DATASET_PATH + '/train/train_label' # All labels are zero in train data.

        data=np.load(train_dataset_path)
        labels=np.load(train_label_file)

        #region=data[:,0]
        #month=data[:,2]

        ### test
        #idx= (month>2)&(month<5)
        #labels=labels[idx]
        #data=data[idx]
        #region=data[:,0]
        #month=data[:,2]
        #############

        
        train_dust=[[[elem[2*i],elem[2*i+1]] for i in range(2,7)] for elem in data]
        train_dust=np.asarray(train_dust)
        #model.fit({'region_input':region,'month_input':month,'dust_input':train_dust},
        #        {'tiny_dense':labels[:,0],'micro_dense':labels[:,1]},
        #        batch_size=128,epochs=EPOCHS,validation_split=0.1)

        train_out=[np.array([[elem[2*i],elem[2*i+1]] for i in range(2,7)]) for elem in data]
        train_out=[np.append(elem,[labels[idx]],0) for idx,elem in enumerate(train_out)]
        train_out=np.asarray(train_out)

        model.fit(train_dust,train_out,
                batch_size=64,epochs=EPOCHS,validation_split=0.1,callbacks=cbs)

        nsml.save(EPOCHS)