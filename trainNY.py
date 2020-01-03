from ST3DNet import *
import pickle
from utils import *
import os
import math
# from keras.utils import plot_model
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

nb_epoch = 1500  # number of epoch at training stage
nb_epoch_cont = 20  # number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 24  # number of time intervals in one day
lr = 0.00002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 10  # divide data into two subsets: Train & Test, of which the test set is the last 10 days
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)

filename = os.path.join("data", 'NYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None
model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)

adam = Adam(lr=lr)
model.compile(loss='mse', optimizer=adam, metrics=[rmse])
model.summary()
# plot_model(model, to_file='model.png',show_shapes=True)

from keras.callbacks import EarlyStopping, ModelCheckpoint
hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
fname_param = '{}.best.h5'.format(hyperparams_name)

early_stopping = EarlyStopping(monitor='val_rmse', patience=50, mode='min')
model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

print('=' * 10)
print("training model...")
history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)

model.save_weights('{}.h5'.format(hyperparams_name), overwrite=True)
print('=' * 10)
print('evaluating using the model that has the best loss on the valid set')
model.load_weights(fname_param)
score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
print('Train score: %.6f  rmse (real): %.6f' %(score[0], score[1] * m_factor))


score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f  rmse (real): %.6f' %(score[0], score[1] * m_factor))

print('=' * 10)
print("training model (cont)...")
fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint], validation_data=(X_test, Y_test))
model.save_weights('{}_cont.h5'.format(hyperparams_name), overwrite=True)

print('=' * 10)
print('evaluating using the final model')
score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
print('Train score: %.6f  rmse (real): %.6f' %
       (score[0], score[1] * m_factor))

score = model.evaluate(
    X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
print('Test score: %.6f  rmse (real): %.6f' %
       (score[0], score[1] * m_factor))