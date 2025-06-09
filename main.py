import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as tkb 
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay


def rmsle(y_true, y_pred):
    y_true = tkb.cast(y_true, 'float32')  
    y_pred = tkb.cast(y_pred, 'float32')  
    
    y_true = tkb.clip(y_true, 0, None) 
    y_pred = tkb.clip(y_pred, 0, None)
    
    return tkb.sqrt(tkb.mean(tkb.square(tkb.log(y_pred + 1) - tkb.log(y_true + 1))))

def split_train_test(train, test):
    # 要拿來當 X 的欄位
    feature_columns = [
    'day_of_week',
    'air_genre_name',
    'air_area_name',
    'latitude',
    'longitude',
    'air_reserve_visitors',
    'air_reserve_count',
    'air_reserve_days_diff_mean',
    'hpg_reserve_visitors',
    'hpg_reserve_count',
    'hpg_reserve_days_diff_mean',
    'holiday_flg',
    'store_mean_visitors',
    'dow_store_mean_visitors',
    'genre_mean_visitors',
    'rolling_mean_3',
    'is_weekend',
    'consecutive_holiday'
    ]

    # 切出訓練資料
    X_train = train[feature_columns]
    y_train = train['visitors']

    # 切出測試資料
    X_test = test[feature_columns]
    y_test = test['visitors']  
    
    return X_train, y_train, X_test, y_test

def label_transform():
    le = sp.LabelEncoder()

    # 對 air_genre_name 進行數值化
    train['air_genre_name'] = le.fit_transform(train['air_genre_name'])
    test['air_genre_name'] = le.transform(test['air_genre_name'])

    # 對 air_area_name 進行數值化
    le = sp.LabelEncoder()  
    train['air_area_name'] = le.fit_transform(train['air_area_name'])
    test['air_area_name'] = le.transform(test['air_area_name'])

    
def visitor_check():
    print(train['visitors'].describe())

    plt.hist(train['visitors'], bins=100)
    plt.xlabel('Number of visitors')
    plt.ylabel('Frequency')
    plt.title('Visitors Distribution')
    plt.show()

#處理air_visit_data.csv

air_visit = pd.read_csv('air_visit_data.csv')
#print(air_visit.head())

air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
#print(air_visit.dtypes)

air_visit['day_of_week'] = air_visit['visit_date'].dt.dayofweek #多一欄星期幾
#print(air_visit.head())

# 2016年的資料  訓練集
train = air_visit[air_visit['visit_date'].dt.year == 2016]

# 2017年的資料  測試集
test = air_visit[air_visit['visit_date'].dt.year == 2017]

#print(f"Train shape: {train.shape}")
#print(f"Test shape: {test.shape}")
# 訓練資料檢查
#print(train.isnull().sum())
# 測試資料檢查
#print(test.isnull().sum())
#visitor_check()

#處理air_store_info.csv

air_store = pd.read_csv('air_store_info.csv')
#print(air_store.head())
#print(air_store.isnull().sum())

# 合併 train 資料
train = pd.merge(train, air_store, how='left', on='air_store_id')

# 合併 test 資料
test = pd.merge(test, air_store, how='left', on='air_store_id')
#print(train.head())
#print(test.head())

#處理air_reserve.csv

air_reserve = pd.read_csv('air_reserve.csv')

air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])

# 多增加預約提前幾天的特徵
air_reserve['reserve_days_diff'] = (air_reserve['visit_datetime'] - air_reserve['reserve_datetime']).dt.days

# 把日期特別抓出來，配合train裡面現在有的日期
air_reserve['visit_date'] = air_reserve['visit_datetime'].dt.date

air_reserve_agg = air_reserve.groupby(['air_store_id', 'visit_date']).agg({
    'reserve_visitors': ['sum', 'count'],
    'reserve_days_diff': 'mean'
}).reset_index()

# 欄位名稱展開（變成一層）
air_reserve_agg.columns = ['air_store_id', 'visit_date', 'air_reserve_visitors', 'air_reserve_count', 'air_reserve_days_diff_mean']
#print(air_reserve_agg.head())

#不轉型態會跳錯誤訊息
air_reserve_agg['visit_date'] = pd.to_datetime(air_reserve_agg['visit_date'])

# 合併
train = pd.merge(train, air_reserve_agg, how='left', on=['air_store_id', 'visit_date'])
test = pd.merge(test, air_reserve_agg, how='left', on=['air_store_id', 'visit_date'])

# 出現一堆NaN，把 NaN 補成 0
train['air_reserve_visitors'].fillna(0, inplace=True)
train['air_reserve_count'].fillna(0, inplace=True)
train['air_reserve_days_diff_mean'].fillna(0, inplace=True)

test['air_reserve_visitors'].fillna(0, inplace=True)
test['air_reserve_count'].fillna(0, inplace=True)
test['air_reserve_days_diff_mean'].fillna(0, inplace=True)

#print(train[['air_store_id', 'visit_date', 'air_reserve_visitors', 'air_reserve_count', 'air_reserve_days_diff_mean']].head())
#print(test[['air_store_id', 'visit_date', 'air_reserve_visitors', 'air_reserve_count', 'air_reserve_days_diff_mean']].head())

#處理hpg_reserve.csv和store_id_relation.csv
hpg_reserve = pd.read_csv('hpg_reserve.csv')
store_id_relation = pd.read_csv('store_id_relation.csv')

hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])

hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].dt.date
hpg_reserve['visit_date'] = pd.to_datetime(hpg_reserve['visit_date'])

hpg_reserve = pd.merge(hpg_reserve, store_id_relation, how='inner', on='hpg_store_id')


hpg_reserve['reserve_days_diff'] = (hpg_reserve['visit_datetime'] - hpg_reserve['reserve_datetime']).dt.days

hpg_reserve_agg = hpg_reserve.groupby(['air_store_id', 'visit_date']).agg({
    'reserve_visitors': ['sum', 'count'],
    'reserve_days_diff': 'mean'
}).reset_index()

hpg_reserve_agg.columns = ['air_store_id', 'visit_date', 'hpg_reserve_visitors', 'hpg_reserve_count', 'hpg_reserve_days_diff_mean']
hpg_reserve_agg['visit_date'] = pd.to_datetime(hpg_reserve_agg['visit_date'])
# 合併
train = pd.merge(train, hpg_reserve_agg, how='left', on=['air_store_id', 'visit_date'])
test = pd.merge(test, hpg_reserve_agg, how='left', on=['air_store_id', 'visit_date'])

train['hpg_reserve_visitors'].fillna(0, inplace=True)
train['hpg_reserve_count'].fillna(0, inplace=True)
train['hpg_reserve_days_diff_mean'].fillna(0, inplace=True)

test['hpg_reserve_visitors'].fillna(0, inplace=True)
test['hpg_reserve_count'].fillna(0, inplace=True)
test['hpg_reserve_days_diff_mean'].fillna(0, inplace=True)

#print(train[['air_store_id', 'visit_date', 'air_reserve_visitors', 'hpg_reserve_visitors']].head())

#處理假日
date_info = pd.read_csv('date_info.csv')

date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])

# 合併
train = pd.merge(train, date_info[['calendar_date', 'holiday_flg']], how='left', left_on='visit_date', right_on='calendar_date')
test = pd.merge(test, date_info[['calendar_date', 'holiday_flg']], how='left', left_on='visit_date', right_on='calendar_date')

# 刪掉多出來的 calendar_date
train.drop('calendar_date', axis=1, inplace=True)
test.drop('calendar_date', axis=1, inplace=True)

#print(train[['visit_date', 'holiday_flg']].head())



# 特徵工程：加入統計與時間特徵（只使用2016資料計算）

# is_weekend 特徵
train['is_weekend'] = train['day_of_week'].isin([5, 6]).astype(int)
test['is_weekend'] = test['day_of_week'].isin([5, 6]).astype(int)

# genre_mean_visitors
genre_mean = train.groupby('air_genre_name')['visitors'].mean().reset_index()
genre_mean.columns = ['air_genre_name', 'genre_mean_visitors']
train = pd.merge(train, genre_mean, on='air_genre_name', how='left')
test = pd.merge(test, genre_mean, on='air_genre_name', how='left')

# store_mean_visitors
store_mean = train.groupby('air_store_id')['visitors'].mean().reset_index()
store_mean.columns = ['air_store_id', 'store_mean_visitors']
train = pd.merge(train, store_mean, on='air_store_id', how='left')
test = pd.merge(test, store_mean, on='air_store_id', how='left')

# dow_store_mean_visitors
dow_store_mean = train.groupby(['air_store_id', 'day_of_week'])['visitors'].mean().reset_index()
dow_store_mean.columns = ['air_store_id', 'day_of_week', 'dow_store_mean_visitors']
train = pd.merge(train, dow_store_mean, on=['air_store_id', 'day_of_week'], how='left')
test = pd.merge(test, dow_store_mean, on=['air_store_id', 'day_of_week'], how='left')

# rolling_mean_3：過去三天的平均（shift+rolling）
train = train.sort_values(by=['air_store_id', 'visit_date'])
train['rolling_mean_3'] = train.groupby('air_store_id')['visitors'].transform(lambda x: x.shift(1).rolling(window=3).mean())
train['rolling_mean_3'].fillna(0, inplace=True)

# 對 test 使用 train 的最後3天來模擬 rolling
rolling_test = train.groupby('air_store_id').tail(3).groupby('air_store_id')['visitors'].mean().reset_index()
rolling_test.columns = ['air_store_id', 'rolling_mean_3']
test = pd.merge(test, rolling_test, on='air_store_id', how='left')
test['rolling_mean_3'].fillna(0, inplace=True)

# consecutive_holiday 特徵
date_info['holiday_flg'] = date_info['holiday_flg'].astype(int)
date_info['consecutive_holiday'] = (
    date_info['holiday_flg'].shift(1, fill_value=0) +
    date_info['holiday_flg'] +
    date_info['holiday_flg'].shift(-1, fill_value=0)
).ge(2).astype(int)

# 合併進來
train = pd.merge(train, date_info[['calendar_date', 'consecutive_holiday']], how='left', left_on='visit_date', right_on='calendar_date')
train.drop('calendar_date', axis=1, inplace=True)

test = pd.merge(test, date_info[['calendar_date', 'consecutive_holiday']], how='left', left_on='visit_date', right_on='calendar_date')
test.drop('calendar_date', axis=1, inplace=True)

label_transform()
#print(train[['air_genre_name', 'air_area_name']].head())

#訓練
X_train, y_train, X_test, y_test = split_train_test(train, test)
# 轉換 y_train 和 y_test
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#print(y_train_log.head())
#a = input()
#print(f"X_train shape: {X_train.shape}")
#print(f"y_train shape: {y_train.shape}")
#print(f"X_test shape: {X_test.shape}")
#print(f"y_test shape: {y_test.shape}")

initial_lr = 5e-4         # 初始學習率
decay_steps = 250         # 多少步內完成一個完整餘弦週期
alpha = 3e-5          # 最小學習率

lr_schedule = CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    alpha=alpha / initial_lr  # alpha 是最終學習率比例 (非絕對值)
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    
    Dense(1)  # 最終輸出，無 activation
])
#model.summary()

#a = input()

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=rmsle,
    metrics=[rmsle]
)
#print("finish")

history = model.fit(
    X_train,
    y_train_log,
    epochs=250,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_rmsle = model.evaluate(X_test, y_test_log, batch_size=512, verbose=1)

##with open("loss_log1p_batch512_Adamlr001_relu.txt", "w") as f:
    ##for epoch_loss in history.history['loss']:
        ##f.write(f"{epoch_loss}\n")

print(f"測試集 Loss (RMSLE, log): {test_loss:.4f}")
print(f"測試集 RMSLE (metric, log): {test_rmsle:.4f}")

plt.figure(figsize=(10,6))

# 畫訓練損失
plt.plot(history.history['loss'], label='Training Loss (RMSLE)')

# 畫驗證損失
plt.plot(history.history['val_loss'], label='Validation Loss (RMSLE)')

# 標題和標籤
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSLE')
plt.legend()

# 顯示圖形
plt.show()
