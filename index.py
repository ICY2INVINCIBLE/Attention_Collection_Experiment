import requests
from gevent import monkey
from flask import Flask, render_template, request,redirect,jsonify
from matplotlib import pyplot as plt
from datetime import timedelta
import define
import xlsxwriter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time,mne,json
import os
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations


#plt.rcParams['mine_font.sans-serif']=['SimHei'] #解决中文乱码
app = Flask(__name__)
app.jinja_env.auto_reload=True
app.config['TEMPLATES_AUTO_RELOAD']=True
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.send_file_max_age_default = timedelta(seconds=1)

PATH='dataset/'
global raw,picks

def get_EEG():
    BoardShim.enable_dev_board_logger()
    # use synthetic board for demo
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    # num_samples=.... 自己设置
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(5)
    data = board.get_board_data()

    #eeg_channels=BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    eeg_channels=[1,2,3,4,5,6,7,8]
    eeg_data=data[eeg_channels,:]

    ch_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']
    ch_names=['TP9','TP10','AF7','AF8','Fp1','Fp2','P3','P4']

    board.stop_stream()
    board.release_session()
    # 把数据保存下来先吧，剩下的之后再处理
    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    print(sfreq)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # 创建raw对象
    global raw,picks
    raw = mne.io.RawArray(eeg_data, info)
    print(raw.info)
    picks = mne.pick_types(
        info, meg=False, eeg=True, stim=False,
        include=ch_names
    )

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    # second eeg channel of synthetic board is a sine wave at 10Hz, should see huge alpha
    eeg_channel = eeg_channels[1]
    # optional detrend
    # 前4个通道
    define.band=[]
    for eeg_channel in range(4):
        DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                                       WindowFunctions.BLACKMAN_HARRIS.value)
        # theta, alpha, beta, gamma
        band_power_alpha = DataFilter.get_band_power(psd, 8.0, 13.0)
        band_power_beta = DataFilter.get_band_power(psd, 13.0, 32.0)
        band_power_gamma = DataFilter.get_band_power(psd, 32.0, 50.0)
        band_power_delta=DataFilter.get_band_power(psd, 0.5, 4.0)
        band_power_theta = DataFilter.get_band_power(psd, 4.0, 8.0)

        print("alpha:%f", band_power_alpha)
        print("beta:%f:", band_power_beta)
        print("gamma:%f", band_power_gamma)
        print("delta:%f", band_power_delta)
        print("theta:%f", band_power_theta)

        define.band.append(band_power_alpha)
        define.band.append(band_power_beta)
        define.band.append(band_power_gamma)
        define.band.append(band_power_delta)
        define.band.append(band_power_theta)

@app.route('/index',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template("index.html")
    else:
        print("request.form=", request.form)
        print(request.form.get("name"))
        print(request.form.get("age"))
        print(request.form.get("sex"))
        print(request.form.get("times"))
        define.SubName=request.form.get("name")
        define.age=request.form.get('age')
        define.sex=request.form.get('sex')
        define.times = int(request.form.get('times'))
        return redirect('calculation')

@app.route('/calculation',methods=['GET','POST'])
def calculation():
    if request.method=='GET':
        return render_template("calculation.html")

@app.route('/algorithm',methods=['GET','POST'])
def algorithm():
    if request.method=='GET':
        define.task=1
        print(define.task)
        threads = [threading.Thread(target=get_EEG)]
        for t in threads:
            # 启动线程
            t.start()
        return render_template('algorithm.html')

@app.route('/MineSweeper',methods=['GET','POST'])
def MineSweeper():
    if request.method=='GET':
        define.task=2
        print(define.task)
        threads = [threading.Thread(target=get_EEG)]
        for t in threads:
            # 启动线程
            t.start()
        return render_template('MineSweeper1.html')


@app.route('/Read',methods=['GET','POST'])
def Read():
    if request.method=='GET':
        define.task=3
        print(define.task)
        threads = [threading.Thread(target=get_EEG)]
        for t in threads:
            # 启动线程
            t.start()
        return render_template('Read.html')


@app.route('/sam',methods=['GET','POST'])
def sam():
    if request.method=='GET':
        return render_template('sam.html')
    else:
        print("request.form=", request.form)
        print("attention",request.form.get('attention'))

        define.attention=request.form.get('attention')
        dirs=PATH+str(define.SubName)+'_'+str(define.age)+'_'+str(define.sex)+'_'+str(define.times)
        if os.path.exists(dirs)==False:
            os.makedirs(PATH+str(define.SubName)+'_'+str(define.age)+'_'+str(define.sex)+'_'+str(define.times))
        dirs=PATH+str(define.SubName)+'_'+str(define.age)+'_'+str(define.sex)+'_'+str(define.times)+'/'
        # keep raw.fif
        raw.save(dirs + str(define.SubName) +'_'+str(define.task )+'_'+str(define.attention) + '_'+str(define.times)+".fif", picks=picks)
        print(dirs + str(define.SubName)+'_'+str(define.task) +'_'+ str(define.attention) +'_'+str(define.times)+ ".fif")
        # keep xlsx
        wookbook=xlsxwriter.Workbook(dirs + str(define.SubName) +'_'+str(define.task )+'_'+str(define.attention) +'_'+str(define.times)+ ".xlsx")
        sheet=wookbook.add_worksheet()
        for i in range(16):
            if (i+1)%4==1:
                sheet.write(0,i,'alpha')
            elif (i+1)%4==2:
                sheet.write(0,i,'beta')
            elif (i+1)%4==3:
                sheet.write(0,i,'gamma')
            else:
                sheet.write(0,i,'theta')
        i=0
        print(define.band)
        for line in define.band:
            sheet.write(1,i,line)
            i+=1
        wookbook.close()

        if define.task==1:
            return redirect('MineSweeper')
        elif define.task==2:
            return redirect('Read')
        else:
            define.times-=1
            if define.times<=0:
                return redirect('end')
            else:
                return redirect('calculation')

@app.route('/end',methods=['GET','POST'])
def end():
    if request.method=='GET':
        return render_template("end.html")

if __name__ == '__main__':
    app.run(port=5050)