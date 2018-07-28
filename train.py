from model import RDN
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.callbacks import TensorBoard,ModelCheckpoint
from datapipeline import get_train_data
import glob
import os
Batch_size = 16
modelsavedir = './modelCP/'
init_lr=1e-4
def scheduler(epoch):
    # lr = K.get_value(myrdn.model.optimizer.lr)
    lr = init_lr * 0.8 **(epoch//50)
    return lr

def gen(x,y):
    while True:
        yield (next(x),next(y))

if __name__ == '__main__':
    myrdn=RDN()
    lr_decay = LearningRateScheduler(scheduler, verbose=1)
    tfboard = TensorBoard()
    modelcp = ModelCheckpoint(modelsavedir+'{epoch:04d}-RDN-{val_loss:.2f}-weights.hdf5', verbose=1, period=1,
                              save_weights_only=True,save_best_only=True)
    gen_tx, gen_ty, gen_vx, gen_vy = get_train_data(Batch_size)
    train_gen=gen(gen_tx,gen_ty)
    valid_gen=gen(gen_vx,gen_vy)
    myrdn.setting_train()

    # 载入之前的模型
    modellist = glob.glob(modelsavedir + '*RDN*.hdf5')
    modellist.sort(key=lambda x: float(x[len(modelsavedir) + 0:len(modelsavedir) + 4]))
    myrdn.load_weight(modellist[-1])
    print('载入', modellist[-1])
    init_epoch = int(os.path.basename(modellist[-1])[:4])
    target_epoch = 170
    step=400
    print('目标',target_epoch,'还有',target_epoch-init_epoch,'\n时间（分）',
          3/(500*24)*step*Batch_size*(target_epoch-init_epoch))

    try:
        myrdn.model.fit_generator(train_gen, step,
                                  epochs=target_epoch, verbose=1,
                            validation_data=valid_gen, validation_steps=6,
                            callbacks=[lr_decay, modelcp],initial_epoch=init_epoch)
    except Exception as e:
        print(e)
    finally:
        myrdn.model.save_weights('weight.h5')
        print('save model')
