import pandas as pd
import numpy as np
import time
from ctgan.ctgan import CTGANSynthesizer


def test_ctgan():
    # ## Credit数据集

    train = pd.read_csv("tests/creditcard.csv")
    train.info()

    # ### 100000条数据

    # print("##########10w条数据##########")
    train_10w= train.sample(1000)
    train_10w.info()
    print(train_10w.shape[1])

    model = CTGANSynthesizer(batch_size=500, epochs=50, verbose=True, amp=True)


    print("----------Train----------")

    st = time.time()
    print("train starting：{}".format(time.strftime("%Y-%m-%d %H:%I:%S", time.localtime(st))))
    model.fit(train_10w)
    et = time.time()
    print("train ending：{}".format(time.strftime("%Y-%m-%d %H:%I:%S", time.localtime(et))))
    print("train time: {} s".format(et-st))
