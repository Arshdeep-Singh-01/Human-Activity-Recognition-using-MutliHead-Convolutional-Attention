import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from utils.Plot.plotInputData import plotData

def readRAWData():
    # path = './Dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.csv'
    path = 'C:/Users/Admin/Desktop/CNN_HAR/Code/utils/Data/Dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.csv'
    file = open(path)
    lines = file.readlines()
    processedList = []
    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            if(len(line)>6):
                continue
            last = line[5].split(';')[0]
            last = last.strip()
            if last == '':
                break
            temp = [line[0], line[1], line[2], line[3], line[4], last]
            processedList.append(temp)
        except:
            print('Error at line number: ', i)

    columns = ['user', 'activity', 'time', 'x', 'y', 'z']
    data = pd.DataFrame(data = processedList, columns = columns)
    return data

def preprocessData(data):
    data['x'] = data['x'].astype('float')
    data['y'] = data['y'].astype('float')
    data['z'] = data['z'].astype('float')

    # plotting input signal graphs
    # plotData(data)

    df = data.drop(['user', 'time'], axis = 1).copy()

    Walking = df[df['activity']=='Walking'].copy()
    Jogging = df[df['activity']=='Jogging'].copy()
    Upstairs = df[df['activity']=='Upstairs'].copy()
    Downstairs = df[df['activity']=='Downstairs'].copy()
    Sitting = df[df['activity']=='Sitting'].copy()
    Standing = df[df['activity']=='Standing'].copy()

    balanced_data = pd.DataFrame()
    balanced_data = balanced_data._append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])

    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['activity'])
    balanced_data.head()

    # standardize data
    X = balanced_data[['x', 'y', 'z']]
    y = balanced_data['label']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_data = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
    scaled_data['label'] = y.values    

    # saving data
    path = 'C:/Users/Admin/Desktop/CNN_HAR/Code/utils/Data/Dataset/WISDM_ar_v1.1/ScaledData.csv'
    scaled_data.to_csv(path, index=False)

    return scaled_data

if __name__ == "__main__":
    data = readRAWData()
    print(data.head())