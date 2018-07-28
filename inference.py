'''
输出预测图像并存起来
'''
from model import RDN
import skimage.io
import glob
import os
import numpy as np
from train import modelsavedir
if __name__ == '__main__':
    #xdir='../Data/Set5_test_LR'
    xdir='../Data/Urban_test_LR'
    xlist=glob.glob(os.path.join(xdir,'*.png'))
    myRDN=RDN()
    #save_dir='./result'
    save_dir='./result2'
    modellist = glob.glob(modelsavedir + '*RDN*.hdf5')
    if len(modellist) > 0:
        modellist.sort(key=lambda x: float(x[len(modelsavedir) + 9:len(modelsavedir) + 13]))
        model=myRDN.load_weight(modellist[0])
        print('载入', modellist[0])
    else:
        model=myRDN.load_weight()
    # 读取图像
    for imgname in xlist:
        print(imgname)
        img=skimage.io.imread(imgname)
        Y=model.predict(np.array([img]),1)[0]
        Y=np.clip(Y,0,255)
        Y=Y.astype(np.uint8)
        skimage.io.imsave(os.path.join(save_dir,os.path.basename(imgname)),Y)