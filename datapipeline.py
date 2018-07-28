'''
使用迭代器送出数据训练
读取返回训练集列表
'''
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator
tx_dir='../Data/tx.npy'
ty_dir='../Data/ty.npy'
vx_dir='../Data/vx.npy'
vy_dir='../Data/vy.npy'
ex_dir='../Data/ex.npy'
ey_dir='../Data/ey.npy'
def get_train_data(Batch_size=16):
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    shuffle=int(time.time()%10000)
    print('randonmint',shuffle)
    vx, vy  = np.load(vx_dir).astype(np.float64), np.load(vy_dir).astype(np.float64)
    tx=np.load(tx_dir).astype(np.float64)
    ty=np.load(ty_dir).astype(np.float64)
    gen_vx = datagen.flow(vx, seed=shuffle, batch_size=1, shuffle=True)
    gen_vy = datagen.flow(vy, seed=shuffle, batch_size=1, shuffle=True)
    gen_tx = datagen.flow(tx, seed=shuffle, batch_size=Batch_size, shuffle=True)
    gen_ty = datagen.flow(ty, seed=shuffle, batch_size=Batch_size, shuffle=True)
    return gen_tx, gen_ty, gen_vx, gen_vy
def get_test_data():
    ex,ey=np.load(ex_dir),np.load(ey_dir)
    return ex,ey