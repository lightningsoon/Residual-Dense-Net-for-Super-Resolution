from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import threading
import random
import cv2
import time
from model import RDN
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
myrdn=RDN()
scale = myrdn.scale  # 缩放倍数
LR_shape = 64
HR_shape = LR_shape * scale
train,test=True,False
n = 100 #测试代码用到参数，全部用None

if bicubic:
    train_x_dir = './DIV2K_train_LR_bicubic/X2' #tx
    valid_x_dir = './DIV2K_valid_LR_bicubic/X2' #vx
else:
    valid_x_dir = './DIV2K_valid_LR_unknown/X2' #vx1
    train_x_dir = './DIV2K_train_LR_unknown/X2' #tx1

train_y_dir = './DIV2K_train_HR' #ty ty1
valid_y_dir = './DIV2K_valid_HR' #vy vy1
def readimg(xl, yl):
    print(xl, yl)
    x, y = image.img_to_array(image.load_img(xl)), \
           image.img_to_array(image.load_img(yl))
    if blur:
        x=cv2.blur(x,(3,3))
    elif Gauss:
        x=cv2.GaussianBlur(x,(5,5),4)

    xs,ys = list(),list()
    shape_x, shape_y = x.shape, y.shape
    if ((shape_x[0] > shape_x[1]) and (shape_y[0] < shape_y[0])) or \
            ((shape_x[0] < shape_x[1]) and (shape_y[0] > shape_y[0])):
        print(xl, yl, '方向不一样，请查看')
        exit(0)
    shape_x, shape_y = (shape_x[0] - LR_shape, shape_x[1] - LR_shape), (shape_y[0] - HR_shape, shape_y[1] - HR_shape)
    for i in range(30):
        row, column = random.randint(0, shape_x[0]), random.randint(0, shape_x[1])
        row1, column1 = row * scale, column * scale
        xx = x[row:row + LR_shape, column:column + LR_shape]
        yy = y[row1:row1 + HR_shape, column1:column1 + HR_shape]
        # print(row1, row1 + LR_shape)
        # print(yy.shape)
        xs.append(xx)
        ys.append(yy)
        # print(np.array(xys).shape)
    # xys=np.array(xys)
    # print(xys.shape)
    # print(len(xys))
    return xs,ys




def data_gen(xdir, ydir):
    # 读取数据
    xlist = os.listdir(xdir)
    ylist = os.listdir(ydir)
    xlist.sort(), ylist.sort()
    # print(xlist,ylist)
    xlist, ylist = list(map(os.path.join, [xdir] * len(xlist), xlist))[:n] \
        , list(map(os.path.join, [ydir] * len(ylist), ylist))[:n]
    print('数量', len(xlist), len(ylist))
    # L=len(xlist)
    assert len(xlist) == len(ylist)
    datax, datay = [], []
    for xl, yl in zip(xlist, ylist):
        temp = readimg(xl, yl)
        datax.extend(temp[0])
        datay.extend(temp[1])
    # print(len(data))
    datax,datay=np.array(datax), np.array(datay)
    print('\n', datax.shape, '\n', '结束一部分\n')
    return datax,datay  # 先转list，再操作


class MyThread(threading.Thread):

    def __init__(self, func, *args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


# for i in range(1):
#     next(gen_vx),next(gen_vy)
#     a = next(gen_vx)#len(a)=16
def getdata():
    shuffle=int(time.time()%10000)
    print('randonmint',shuffle)
    if 'bicubic' in train_x_dir:
        vx, vy  = np.load('vx.npy'), np.load('vy.npy')
        if blur:
            tx = np.load('txB.npy')
            ty = np.load('tyN.npy')
        elif Gauss:
            tx = np.load('txG.npy')
            ty = np.load('tyN.npy')
        else:
            tx=np.load('tx.npy')
            ty=np.load('ty.npy')
    if 'unknown' in train_x_dir:
        vx, vy = np.load('vx1.npy'), np.load('vy1.npy')
        if blur:
            tx = np.load('tx1B.npy')
            ty = np.load('tyN.npy')
        elif Gauss:
            tx = np.load('tx1G.npy')
            ty = np.load('tyN.npy')
        else:
            tx=np.load('tx1.npy')
            ty = np.load('ty1.npy')
    # gen_vx=datagen.flow(vx,seed=1,batch_size=Batch_sizes,shuffle=False,save_to_dir='valid_x')
    # gen_vy=datagen.flow(vy,seed=1,batch_size=Batch_sizes,shuffle=False,save_to_dir='valid_y')
    gen_vx = datagen.flow(vx, seed=shuffle, batch_size=Batch_sizes, shuffle=True)
    gen_vy = datagen.flow(vy, seed=shuffle, batch_size=Batch_sizes, shuffle=True)

    # gen_tx=datagen.flow(tx,seed=1,batch_size=Batch_sizes,shuffle=True,save_to_dir='train_x')不要轻易使用，会占用大量空间
    # gen_ty=datagen.flow(ty,seed=1,batch_size=Batch_sizes,shuffle=True,save_to_dir='train_y')
    gen_tx = datagen.flow(tx, seed=shuffle, batch_size=Batch_sizes, shuffle=True)
    gen_ty = datagen.flow(ty, seed=shuffle, batch_size=Batch_sizes, shuffle=True)
    return gen_tx, gen_ty, gen_vx, gen_vy


if __name__ == '__main__':
    # 先保存一下npy，再加载就快多了
    if train:
        mythreadx = MyThread(data_gen, train_x_dir, train_y_dir)
        mythreadx.setDaemon(True)
        mythreadx.start()
    if test:
        mythready = MyThread(data_gen, valid_x_dir, valid_y_dir)
        mythready.setDaemon(True)
        mythready.start()

    # time.sleep(20)
    if test:
        mythready.join()
        vx, vy = mythready.get_result()
        # print('vxshape', vx.shape)
        if bicubic:

            np.save('vx.npy', vx)
            np.save('vy.npy', vy)
        else:
            np.save('vx1.npy', vx)
            np.save('vy1.npy', vy)
        print('save v')  # (1600,2)
    if train:
        mythreadx.join()
        tx, ty = mythreadx.get_result()
        if bicubic:
            if blur:
                np.save('txB.npy',tx)
                np.save('tyN.npy',ty)
            elif Gauss:
                np.save('txG.npy',tx)
                np.save('tyN.npy',ty)
            else:
                np.save('tx.npy', tx)
                np.save('ty.npy', ty)
        else:
            if blur:
                np.save('tx1B.npy',tx)
                np.save('ty1N.npy',ty)
            elif Gauss:
                np.save('tx1G.npy',tx)
                np.save('ty1N.npy',ty)
            else:
                np.save('tx1.npy',tx)
                np.save('ty1.npy', ty)
        print('save t')  # (12800, 2)

