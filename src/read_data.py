import pandas as pd
import gc


def read_data():
    # read total exposure log.out
    col_name = ['id', 'request_timestamp', 'position', 'uid', 'aid', 'imp_ad_size',
                'bid', 'pctr', 'quality_ecpm', 'totalEcpm']
    df = pd.read_csv('data/totalExposureLog.out', sep='\t', names=col_name,
                     engine='python', iterator=True)
    loop = True
    chunk_size = 1000000
    chunks = []
    index = 0
    while loop:
        try:
            # convert dtypes
            chunk = df.get_chunk(chunk_size)
            chunk[['id', 'request_timestamp', 'position', 'uid', 'aid', 'imp_ad_size']] = \
                chunk[['id', 'request_timestamp', 'position', 'uid', 'aid', 'imp_ad_size']].astype(int)
            chunk[['bid', 'pctr', 'quality_ecpm', 'totalEcpm']] = \
                chunk[['bid', 'pctr', 'quality_ecpm', 'totalEcpm']].astype(float)
            # drop missing value
            chunk = chunk.dropna()
            # remove duplicated
            chunk = chunk.drop_duplicates(keep='first', inplace=False)
            chunk = chunk.reset_index(drop=True)
            chunks.append(chunk)
            index += 1

        except StopIteration:
            loop = False
            print('Iteration is stopped.')

    df = pd.concat(chunks, ignore_index=True).sort_values(by='request_timestamp')
    # serialize data
    df.to_pickle('data/totalExposureLog.pkl')
    del df
    gc.collect()
    print('Finish totalExposureLog.pkl')

    # read ad_static_feature
    col_name = ['aid', 'create_timestamp', 'advertiser', 'good_id', 'good_type', 'ad_type_id', 'ad_size']
    df = pd.read_csv('data/ad_static_feature.out', sep='\t', names=col_name).sort_values(by='create_timestamp')
    df = df.fillna(-1)
    for f in ['aid', 'create_timestamp', 'advertiser', 'good_id', 'good_type', 'ad_type_id']:
        items = []
        for item in df[f].values:
            try:
                items.append(int(item))
            except:
                items.append(-1)
        df[f] = items
        df[f] = df[f].astype(int)
    df['ad_size'] = df['ad_size'].apply(lambda x: ' '.join([str(int(float(y))) for y in str(x).split(',')]))
    df.to_pickle('data/ad_static_feature.pkl')
    del df
    gc.collect()
    print('Finish ad_static_feature.pkl')


    # read user data
    col_name = ['uid', 'age', 'gender', 'area', 'status', 'education', 'concuptionAbility', 'os', 'work',
                'connectionType', 'behavior']
    df = pd.read_csv('data/user_data', sep='\t', names=col_name)
    df = df.fillna(-1)
    df[['uid', 'age', 'gender', 'education', 'consuptionAbility', 'os', 'connectionType']] = df[
        ['uid', 'age', 'gender', 'education', 'concuptionAbility', 'os', 'connectionType']].astype(int)
    for f in ['area', 'status', 'work', 'behavior']:
        df[f] = df[f].apply(lambda x: ' '.join(x.split(',')))
    df.to_pickle('data/user_data.pkl')
    del df
    gc.collect()
    print('Finish user_data.pkl')

    # read test data
    col_name = ['id', 'aid', 'create_timestamp', 'ad_size', 'ad_type_id', 'good_type', 'good_id', 'advertiser',
                'delivery_periods', 'crowd_direction', 'bid']
    df = pd.read_csv('data/test_sample.dat', sep='\t', names=col_name)
    df = df.fillna(-1)
    df[['id', 'aid', 'create_timestamp', 'ad_size', 'ad_type_id', 'good_type', 'good_id', 'advertiser']] = df[
        ['id', 'aid', 'create_timestamp', 'ad_size', 'ad_type_id', 'good_type', 'good_id', 'advertiser']].astype(int)
    df['bid'] = df['bid'].astype(float)
    df.to_pickle('data/test_sample.pkl')
    del df
    gc.collect()
    print('Finish test_sample.pkl')


print('Reading raw data: ')
read_data()
