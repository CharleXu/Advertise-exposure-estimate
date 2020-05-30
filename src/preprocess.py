import os
import pandas as pd
import numpy as np
import random
import gc
import time
from tqdm import tqdm


def create_dataset():
    train_df = pd.read_pickle('data/totalExposureLog.pkl')
    train_df['request_day'] = train_df['request_timestamp'] // (3600 * 24)
    weekday = []
    hour = []
    minute = []
    for x in tqdm(train_df['request_timestamp'].values, total=len(train_df)):
        localtime = time.localtime(x)
        weekday.append(localtime[6])
        hour.append(localtime[3])
        minute.append(localtime[4])
    train_df['weekday'] = weekday
    train_df['hour'] = hour
    train_df['minute'] = minute
    train_df['period_id'] = train_df['hour'] * 2 + train_df['minute'] // 30
    dev_df = train_df[train_df['request_day'] == 17974]
    del dev_df['period_id']
    del dev_df['minute']
    del dev_df['hour']
    log = train_df
    tmp = pd.DataFrame(train_df.groupby(['aid', 'request_day']).size()).reset_index()
    tmp.columns = ['aid', 'request_day', 'imp']
    log = log.merge(tmp, on=['aid', 'request_day'], how='left')
    log[log['request_day'] < 17973].to_pickle('data/user_log_dev.pkl')
    log.to_pickle('data/user_log_test.pkl')
    del log
    del tmp
    gc.collect()
    del train_df['period_id']
    del train_df['minute']
    del train_df['hour']
    return train_df, dev_df


# process ad_operation data
def extract_setting():
    aids = []
    with open('data/ad_operation.dat', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            try:
                if line[1] == '20190230000000':
                    line[1] = '20190301000000'
                if line[1] != '0':
                    request_day = time.mktime(time.strptime(line[1], '%Y%m%d%H%M%S')) // (3600 * 24)
                else:
                    request_day = 0
            except:
                print(line[1])

            if len(aids) == 0:
                aids.append([int(line[0]), 0, "NaN", "NaN"])
            elif aids[-1][0] != int(line[0]):
                for i in range(max(17930, aids[-1][1] + 1), 17975):
                    aids.append(aids[-1].copy())
                    aids[-1][1] = i
                aids.append([int(line[0]), 0, "NaN", "NaN"])
            elif request_day != aids[-1][1]:
                for i in range(max(17930, aids[-1][1] + 1), int(request_day)):
                    aids.append(aids[-1].copy())
                    aids[-1][1] = i
                aids.append(aids[-1].copy())
                aids[-1][1] = int(request_day)
            if line[3] == '3':
                aids[-1][2] = line[4]
            if line[3] == '4':
                aids[-1][3] = line[4]
    ad_df = pd.DataFrame(aids)
    ad_df.columns = ['aid', 'request_day', 'crowd_direction', 'delivery_periods']
    return ad_df


def create_train_data(train_df):
    # calculate average aid and exposure
    tmp = pd.DataFrame(train_df.groupby(['aid', 'request_day'])['bid'].nunique()).reset_index()
    tmp.columns = ['aid', 'request_day', 'bid_unique']
    train_df = train_df.merge(tmp, on=['aid', 'request_day'], how='left')
    tmp = pd.DataFrame(train_df.groupby(['aid', 'request_day']).size()).reset_index()
    tmp_1 = pd.DataFrame(train_df.groupby(['aid', 'request_day'])['bid'].mean()).reset_index()
    tmp.columns = ['aid', 'request_day', 'imp']
    del train_df['bid']
    tmp_1.columns = ['aid', 'request_day', 'bid']
    train_df = train_df.drop_duplicates(['aid', 'request_day'])
    train_df = train_df.merge(tmp, on=['aid', 'request_day'], how='left')
    train_df = train_df.merge(tmp_1, on=['aid', 'request_day'], how='left')
    del tmp
    del tmp_1
    gc.collect()
    # drop duplicates
    train_df = train_df.drop_duplicates(['aid', 'request_day'])
    del train_df['request_timestamp']
    del train_df['uid']

    # remove missing value
    ad_df = extract_setting()
    ad_df = ad_df.drop_duplicates(['aid', 'request_day'], keep='last')
    ad_df['request_day'] += 1
    train_df = train_df.merge(ad_df, on=['aid', 'request_day'], how='left')
    train_df['is'] = train_df['crowd_direction'].apply(lambda x: type(x) == str)
    train_df = train_df[train_df['is'] == True]
    train_df = train_df[train_df['crowd_direction'] != "NaN"]
    train_df = train_df[train_df['delivery_periods'] != "NaN"]

    # remove outliers
    train_df = train_df[train_df['imp'] <= 3000]
    train_df = train_df[train_df['bid'] <= 1000]
    train_dev_df = train_df[train_df['request_day'] < 17973]
    print(train_df.shape, train_dev_df.shape)
    print(train_df['imp'].mean(), train_df['bid'].mean())
    return train_df, train_dev_df


def create_dev_data(dev_df):
    # create validation dataset
    aids = set()
    exit_aids = set()
    with open('data/ad_operation.dat', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[1] == '20190230000000':
                line[1] = '20190301000000'
            if line[1] != '0':
                request_day = time.mktime(time.strptime(line[1], '%Y%m%d%H%M%S')) // (3600 * 24)
            else:
                request_day = 0
            if request_day == 17974:
                aids.add(int(line[0]))
            exit_aids.add(int(line[0]))
    dev_df['is'] = dev_df['aid'].apply(lambda x: x in aids)
    dev_df = dev_df[dev_df['is'] == False]
    dev_df['is'] = dev_df['aid'].apply(lambda x: x in exit_aids)
    dev_df = dev_df[dev_df['is'] == True]
    # filter non-unique aid
    tmp = pd.DataFrame(dev_df.groupby('aid')['bid'].nunique()).reset_index()
    tmp.columns = ['aid', 'bid_unique']
    dev_df = dev_df.merge(tmp, on='aid', how='left')
    dev_df = dev_df[dev_df['bid_unique'] == 1]
    # calculate exposure
    tmp = pd.DataFrame(dev_df.groupby('aid').size()).reset_index()
    tmp.columns = ['aid', 'imp']
    dev_df = dev_df.merge(tmp, on='aid', how='left')
    dev_df = dev_df.drop_duplicates('aid')
    # remove missing value
    ad_df = extract_setting()
    ad_df = ad_df.drop_duplicates(['aid'], keep='last')
    dev_df = dev_df.merge(ad_df, on='aid', how='left')
    dev_df = dev_df[dev_df['crowd_direction'] != "NaN"]
    dev_df = dev_df[dev_df['delivery_periods'] != "NaN"].reset_index()
    del dev_df['index']
    del dev_df['request_timestamp']
    del dev_df['is']
    del dev_df['uid']
    # create fake data to test
    items = []
    for item in dev_df[['aid', 'bid', 'crowd_direction', 'delivery_periods', 'imp']].values:
        item = list(item)
        items.append(item + [1])
        for i in range(10):
            while True:
                t = random.randint(0, 2 * item[1])
                if t != item[1]:
                    items.append(item[:1] + [t] + item[2:] + [0])
                    break
                else:
                    continue
    dev_df = pd.DataFrame(items)
    dev_df.columns = ['aid', 'bid', 'crowd_direction', 'delivery_periods', 'imp', 'gold']
    del items
    gc.collect()
    print(dev_df.shape)
    print(dev_df['imp'].mean(), dev_df['bid'].mean())
    return dev_df


print("construct log ....")
train_df, dev_df = create_dataset()

print("construct train data ....")
train_df, train_dev_df = create_train_data(train_df)

print("construct dev data ....")
dev_df = create_dev_data(dev_df)

print("load test data ....")
test_df = pd.read_pickle('data/test_sample.pkl')

print("combine advertise features ....")
ad_df = pd.read_pickle('data/ad_static_feature.pkl')
train_df = train_df.merge(ad_df, on='aid', how='left')
train_dev_df = train_dev_df.merge(ad_df, on='aid', how='left')
dev_df = dev_df.merge(ad_df, on='aid', how='left')

print("save preprocess data ....")
train_dev_df.to_pickle('data/train_dev.pkl')
train_df.to_pickle('data/train.pkl')
dev_df.to_pickle('data/dev.pkl')
test_df.to_pickle('data/test.pkl')
print(train_dev_df.shape, dev_df.shape)
print(train_df.shape, test_df.shape)
