from keras import backend as K

def PSNR(y_true, y_pred):
    # 只能给255灰度级图像用
    target_value=10*K.log(K.square(255.)/K.mean(K.square(y_pred - y_true)))/K.log(10.)
    return target_value

