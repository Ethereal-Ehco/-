import wfdb
import pywt
import matplotlib.pyplot as plt
'''
    sampfrom: 设置读取心电信号的 起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的 结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    channel_names：设置设置读取心电信号名字，必须是列表，channel_names=['MLII']表示读取MLII导联线
    physical : bool, optional；如果为True则读取p_signal，如果为False则读取d_signal，默认为False；
    channels：设置读取第几个心电信号，必须是列表，channels=[0, 3]表示读取第0和第3个信号，注意信号数不确定

'''
#读取数据
record=wfdb.rdrecord(r'D:\毕业设计\测试\data\MIT-BIH\202',physical=True,channels=[0,])
'''
p_signal：模拟信号值，储存形式为ndarray或者是list（在_signal.py中定义的）

d_signal：数字信号值，储存形式为ndarray或者是list（在_signal.py中定义的）

fs：采样频率，int类型的；
'''
print('record frequency:'+str(record.fs))
ventricular_signal=record.p_signal
print('signal shape:'+str(ventricular_signal.shape))
#plt.plot(ventricular_signal)
#plt.title('ventricular_signal')
#读取标注
signal_annotation = wfdb.rdann("data/MIT-BIH/202", "atr")
# 将读取到的annatations的心拍绘制到心电图上
symbol=signal_annotation.symbol
#print(list(zip(signal_annotation.sample,symbol)))

#for index in signal_annotation.sample:
#    plt.scatter(index, ventricular_signal[index], marker="*")
#plt.show()
      
'''

chan：是chanel的意思，保存的是当前标注的是哪一个通道，为一个ndarray或者是list

sample：这里记录的是每个心拍中R波的位置信息，为一个ndarray或者是list

symbol：记录的是每个心拍的标注信息，记录的是每个心拍的类型，是一个字符型的ndarray或者是list，

        其内容为wfdb.show_ann_labels()的symbol类型
        
aux_note：记录的是心率变换点的标注类型，是辅助信息

'''
ecg=ventricular_signal[0:1000]  # 生成心电信号

index = []
data = []
for i in range(len(ecg)-1):
    X = float(i)
    Y = float(ecg[i])
    index.append(X)
    data.append(Y)


# Create wavelet object and define parameters
w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
print("maximum level is " + str(maxlev))
threshold = 0.04  # Threshold for filtering

# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

mintime = 0
maxtime = mintime + len(data) + 1
'''
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime-1])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()

'''