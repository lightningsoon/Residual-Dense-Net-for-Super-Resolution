from keras.layers import Input, Conv2D, Activation, Concatenate, Add
from keras.utils import plot_model
from tools import PSNR
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Lambda
class RDN():
    def __init__(self, input_shape=(None, None)):
        self.G0 = 64  # 除残差密集模块中的卷积层外的层滤波器个数
        # Shallow feature extraction layers, local and global feature fusion layers have G0=64 filters.
        self.G = 64  # 增长比率，残差密集模块3x3核的输出个数
        self.D = 16  # the number of RDB 模块个数
        self.C = 8  # he number of Conv layers per RDB 模块中卷积层的个数
        self.scale = 2  # 放大倍数
        self.out_channel = 3  # 输出渠道
        self.model = self.get_model(input_shape)
        pass

    def RDB_cn(self):
        # 残差密集模块3x3核
        return Conv2D(self.G,
                      kernel_size=(3, 3),
                      padding='same',
                      activation='relu',
                      )

    def get_model(self, shape):
        if len(shape) == 1:
            LR = Input((shape, shape, 3))
        elif len(shape) == 2:
            LR = Input((*shape, 3))
        elif len(shape) == 3:
            LR = Input(shape)
        else:
            print(shape)
            raise ValueError('dim of {0} must be <= 3'.format(shape))
        # Shallow feature extraction layers
        # SFEL
        prev = self.cn_G0()(LR)
        DFF0 = prev
        prev = self.cn_G0()(prev)
        # residual dense blocks (RDBs)
        DFF = list()
        for i in range(self.D):
            prev = self.RDB(prev)
            DFF.append(prev)
        # dense feature fusion(DFF)功能：全局特征学习
        prev = Concatenate()(DFF)
        prev = self.cn_G0((1, 1))(prev)  # 全局特征融合
        prev = self.cn_G0()(prev)  # 全局特征融合
        prev = Add()([DFF0, prev])  # 全局残差学习
        # ESPCNN 上采样
        prev = self.ESPCNN(prev)
        prev = Conv2D(self.out_channel, (3, 3), padding='same')(prev)
        HR = prev
        model = Model(inputs=[LR], outputs=[HR])
        return model

    def cn_G0(self, size=(3, 3)):
        return Conv2D(self.G0, size, padding='same')

    def RDB(self, x):
        x0 = x
        result = list()
        result.append(x0)  # [in]
        for i in range(self.C):
            x = self.RDB_cn()(x)
            result.append(x)  # [in,RDB_cn]
            x = Concatenate()(result)
        x = self.cn_G0((1, 1))(x)  # feauture fusion 特征融合层
        x = Add()([x0, x])
        return x

    def ESPCNN(self, x):
        x = Conv2D(self.scale ** 2 * self.G, (3, 3),  padding='same')(x)
        # print(x)
        # x = SubPixelConvolution(self.scale)(x)
        # print(x)
        PixelShuffle=Lambda(lambda x: tf.depth_to_space(x,self.scale,name='PixelShuffle'))
        x = PixelShuffle(x)
        return x

    def save_model(self, plot=True, storage=False):
        if plot:
            plot_model(self.model, 'RDN_simple.png', False)
            print('画了图')
        if storage:
            self.model.save('model0.hdf5')
            print('存了模型原始文件')

    def setting_train(self):
        opt = Adam(1e-4)
        loss = 'mae'
        self.model.compile(opt, loss, metrics=[PSNR])
    def load_weight(self,name='weight.h5'):
        self.model.load_weights(name)
        return self.model


if __name__ == '__main__':
    myrdn = RDN()
    myrdn.save_model()
