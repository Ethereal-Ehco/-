import wfdb
import pywt
import matplotlib.pyplot as plt

'''
读取数据进行小波变换后储存到processedData文件夹中
'''
def getData():
    for dataNum in range(230,235):
        path='D:\\毕业设计\\测试\\data\\MIT-BIH\\'+str(dataNum)
        record = wfdb.rdrecord(path, physical=True, channels=[0, ])
        ventricular_signal = record.p_signal

        ecg = ventricular_signal  # 生成心电信号
        data = []
        for i in range(len(ecg) - 1):
            Y = float(ecg[i])
            data.append(Y)

        # Create wavelet object and define parameters
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        print("maximum level is " + str(maxlev))
        threshold = 0.04  # Threshold for filtering

        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

        datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
        #提取R波
        signal_annotation = wfdb.rdann("data/MIT-BIH/202", "atr")
        # 将读取到的annatations的心拍绘制到心电图上
        symbol = signal_annotation.symbol
        plt.figure()
        plt.plot(datarec)
        for index in symbol:
            plt.scatter(index, ventricular_signal[index], marker="*")
        plt.show()
        f=open('D:\\毕业设计\\测试\\data\\processedData\\'+str(dataNum)+'.txt','w')
        for each in datarec:
            f.write(str(each)+'\n')
        f.close()
def test():
    path = 'D:\\毕业设计\\测试\\data\\MIT-BIH\\100'
    record = wfdb.rdrecord(path, physical=True, channels=[0, ])
    ventricular_signal = record.p_signal

    ecg = ventricular_signal  # 生成心电信号
    data = []
    for i in range(len(ecg) - 1):
        Y = float(ecg[i])
        data.append(Y)

    # Create wavelet object and define parameters
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))
    threshold = 0.04  # Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    f = open('D:\\毕业设计\\测试\\data\\processedData\\100' + '.txt', 'w')
    for each in datarec:
        f.write(str(each)+'\n')
    f.close()
def testRwave():
    path = 'D:\\毕业设计\\测试\\data\\MIT-BIH\\' + '100'
    record = wfdb.rdrecord(path, physical=True, channels=[0, ])
    ventricular_signal = record.p_signal
    ecg = ventricular_signal[0:1000]  # 生成心电信号
    plt.figure()
    plt.plot(ecg)
    data = []
    for i in range(len(ecg) - 1):
        Y = float(ecg[i])
        data.append(Y)

    # Create wavelet object and define parameters
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))
    threshold = 0.04  # Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    plt.plot(datarec)
        # 提取R波
    signal_annotation = wfdb.rdann("data/MIT-BIH/100", "atr")
        # 将读取到的annatations的心拍绘制到心电图上
    sample = signal_annotation.sample
    for i in range(len(sample)):
        if sample[i]>1000:
            sample=sample[0:i]
            break
    print(sample)
    for index in sample:
        plt.scatter(index, datarec[index], marker="*")
    plt.show()



if __name__=='__main__':
    testRwave()



