import os
import pandas as pd
import numpy as np

'''
Подрузумевается, что мы находимся в последнем дне обучающего датасета (т.е. в день 1268)
И знаем все, что было до этого дня.

В случае риалтайм-системы фичи следует рассчитывать честно на каждый день 
(то есть на 5 сентября собираем с начала до 4, на 6 сентября с начала до 5 и т.д.).
'''

def collect_events(start_day, end_day):
    all_events = []
    events_dir = os.path.join('t_ecd_small_partial/dataset/small/retail/events/')
    for day in range(start_day, end_day + 1):
        file_path = os.path.join(events_dir, f'0{day}.pq')
        if os.path.exists(file_path):
            events = pd.read_parquet(file_path)
            events['day'] = day
            all_events.append(events)
    data = pd.concat(all_events, ignore_index=True)
    events = data.groupby(['day', 'user_id', 'item_id', 'subdomain', 'os', 'action_type'], as_index=False).size().rename(columns={'size': 'cnt'})
    return events

def collect_reviews(dp, start_day, end_day):
    all_reviews = []
    reviews_dir = os.path.join('t_ecd_small_partial/dataset/small/reviews/')
    for day in range(start_day, end_day + 1):
        file_path = os.path.join(reviews_dir, f'0{day}.pq')
        if os.path.exists(file_path):
            reviews = pd.read_parquet(file_path, engine='fastparquet')
            reviews['day'] = day
            all_reviews.append(reviews)
    reviews = pd.concat(all_reviews, ignore_index=True)
    reviews['user_id'] = reviews['user_id'].apply(lambda x: dp.user_to_idx[x] if x in dp.user_to_idx else None)
    reviews.dropna(subset=['user_id'], inplace=True)
    reviews['user_id'] = reviews['user_id'].astype(np.int64)
    return reviews

def collect_user_features(df, dp, start_day=1082, end_day=1268):
    users = pd.read_parquet('t_ecd_small_partial/dataset/small/users.pq', engine='fastparquet')
    users = users.rename(columns={'user_id': 'original_user_id', 'socdem_cluster': 'user_socdem_cluster', 'region': 'user_region'})
    users['user_id'] = users['original_user_id'].apply(lambda x: dp.user_to_idx[x] if x in dp.user_to_idx else None)
    users.dropna(subset=['user_id'], inplace=True)
    users['user_region'] = users['user_region'].fillna(users['user_region'].mode().iloc[0])

    users_df = df.groupby('user_id').agg(
        first_purchase_day=('day', 'min'),
        last_purchase_day=('day', 'max'),
        user_os=('os', lambda x: x.mode().iloc[0]),
        user_orders = ('user_id', 'count'),
        user_avg_check = ('item_price', 'mean'),
        # user_check_std = ('item_price', 'std'),
        user_total_revenue = ('item_price', 'sum'),
        user_main_subdomain = ('subdomain', lambda x: x.mode().iloc[0]),
        user_brands = ('item_brand_id', 'nunique'),
        user_categories = ('item_category', 'nunique'),
        user_items = ('item_id', 'nunique'),
    ).reset_index()
    users_df = users_df.merge(users, left_on='user_id', right_on='user_id', how='left')

    users_df['user_days_since_first_purchase'] = end_day - users_df['first_purchase_day']
    users_df['user_days_since_last_purchase'] = end_day - users_df['last_purchase_day']
    users_df['user_lifetime'] = users_df['last_purchase_day'] - users_df['first_purchase_day']
    users_df.drop(columns=['first_purchase_day', 'last_purchase_day', 'original_user_id'], inplace=True)

    return users_df

def aggr_events(events, dp, keys=['item_id']):
    events_aggr = events.groupby(keys + ['action_type'], as_index=False)['cnt'].sum()
    if 'item_id' in keys:
        events_aggr['item_id'] = events_aggr['item_id'].apply(lambda x: dp.item_to_idx[x] if x in dp.item_to_idx else None)
        events_aggr = events_aggr.dropna(subset=['item_id'])
        events_aggr['item_id'] = events_aggr['item_id'].astype(np.int64)
    if 'user_id' in keys:
        events_aggr['user_id'] = events_aggr['user_id'].apply(lambda x: dp.user_to_idx[x] if x in dp.user_to_idx else None)
        events_aggr = events_aggr.dropna(subset=['user_id'])
        events_aggr['user_id'] = events_aggr['user_id'].astype(np.int64)
    events_aggr=events_aggr.pivot_table(index=keys, columns='action_type', values='cnt', aggfunc='sum')
    events_aggr = pd.DataFrame(events_aggr.to_records()).fillna(0)
    events_aggr.rename(columns={'added-to-cart': 'item_added_to_cart_total', 'click': 'item_clicked_total', 'view': 'item_viewed_total'}, inplace=True)
    return events_aggr

def collect_item_features(df, dp, start_day=1082, end_day=1268):
    items = pd.read_parquet('t_ecd_small_partial/dataset/small/retail/items.pq', engine='fastparquet')
    items = items.rename(columns={'item_id': 'original_item_id'})
    items['item_id'] = items['original_item_id'].apply(lambda x: dp.item_to_idx[x] if x in dp.item_to_idx else None)
    items = items.drop(columns=['original_item_id'])
    items = items.dropna(subset=['item_id'])
    items['item_id'] = items['item_id'].astype(np.int64)

    # nan's in category
    items['category'] = items['category'].fillna('Unknown category')
    # nan's in subcategory
    items['subcategory'] = items['subcategory'].fillna('Unknown subcategory')
    # nan's in price
    items['price'] = items.groupby(['category', 'subcategory'])['price'].transform(
        lambda x: x.fillna(x.median())
    )
    # если в подгруппе все значения nan
    items['price'] = items.groupby(['category'])['price'].transform(
        lambda x: x.fillna(x.median())
    )

    brands = pd.read_parquet('t_ecd_small_partial/dataset/small/brands.pq', engine='fastparquet')
    # fill nan's
    available_embeddings = brands['embedding'].dropna().tolist()
    mean_brand_embedding = np.mean(np.stack(available_embeddings), axis=0)
    brands['embedding'] = brands['embedding'].apply(
        lambda x: x if x is not None else mean_brand_embedding.tolist()
    )
    # aggregate embeddings
    brands = brands.groupby('brand_id')['embedding'].apply(
        lambda x: np.mean(np.stack(x.tolist()), axis=0).tolist() if len(x) > 0 else None
    ).reset_index()

    brands.columns = ['brand_id', 'brand_embedding']

    items = items.merge(brands, on='brand_id')
    items = items.rename(columns={
        'category': 'item_category',
        'subcategory': 'item_subcategory',
        'price': 'item_price',
        'embedding': 'item_embedding',
        'brand_embedding': 'item_brand_embedding'
        })
    items['item_relative_category_price'] = items['item_price'] / items.groupby('item_category')['item_price'].transform('mean')
    items['item_deviation_category_price'] = items['item_price'] - items.groupby('item_category')['item_price'].transform('mean')

    events = collect_events(start_day, end_day)
    events_aggr_total = aggr_events(events, dp)
    events_aggr_3d = aggr_events(events[events['day'] >= end_day - 3], dp)
    events_aggr_3d.rename(columns={
        'item_added_to_cart_total': 'item_added_to_cart_3d',
        'item_clicked_total': 'item_clicked_3d',
        'item_viewed_total': 'item_viewed_3d'
        }, inplace=True)
    events_aggr = events_aggr_total.merge(events_aggr_3d, on='item_id', how='left').fillna(0)

    events_aggr_7d = aggr_events(events[events['day'] >= end_day - 7], dp)
    events_aggr_7d.rename(columns={
        'item_added_to_cart_total': 'item_added_to_cart_7d',
        'item_clicked_total': 'item_clicked_7d',
        'item_viewed_total': 'item_viewed_7d'
        }, inplace=True)
    events_aggr = events_aggr_total.merge(events_aggr_7d, on='item_id', how='left').fillna(0)


    events_aggr_30d = aggr_events(events[events['day'] >= end_day - 30], dp)
    events_aggr_30d.rename(columns={
        'item_added_to_cart_total': 'item_added_to_cart_30d',
        'item_clicked_total': 'item_clicked_30d',
        'item_viewed_total': 'item_viewed_30d'
        }, inplace=True)
    events_aggr = events_aggr_total.merge(events_aggr_30d, on='item_id', how='left').fillna(0)

    items = items.merge(events_aggr, on='item_id', how='left').fillna(0)

    uniq_users_total = df.groupby(['item_id'], as_index=False).agg(
        item_uniq_users_total = ('user_id', 'nunique')
    )
    uniq_users_3d = df[df['day'] >= end_day - 3].groupby(['item_id'], as_index=False).agg(
        item_uniq_users_3d = ('user_id', 'nunique')
    )
    uniq_users_7d = df[df['day'] >= end_day - 7].groupby(['item_id'], as_index=False).agg(
        item_uniq_users_7d = ('user_id', 'nunique')
    )
    uniq_users_30d = df[df['day'] >= end_day - 30].groupby(['item_id'], as_index=False).agg(
        item_uniq_users_30d = ('user_id', 'nunique')
    )

    items = items.merge(uniq_users_total, on='item_id', how='left').merge(uniq_users_3d, on='item_id', how='left').merge(uniq_users_7d, on='item_id', how='left').merge(uniq_users_30d, on='item_id', how='left').fillna(0)
    return items


def collect_user_brand_features(df, dp, start_day=1082, end_day=1268):
    user_brand = df.groupby(['user_id', 'item_brand_id'], as_index=False).agg(
        user_brand_cnt_orders = ('timestamp', 'count'),
        first_order_day = ('day', 'min'),
        last_order_day = ('day', 'max'),
        user_brand_avg_price = ('item_price', 'mean'),
        user_brand_min_price = ('item_price', 'min'),
        user_brand_max_price = ('item_price', 'max'),
        user_brand_std_price = ('item_price', 'std'),
        
    )
    user_brand['user_brand_days_since_first_order'] = end_day - user_brand['first_order_day']
    user_brand['user_brand_days_since_last_order'] = end_day - user_brand['last_order_day']
    user_brand['user_brand_lifetime'] = user_brand['last_order_day'] - user_brand['first_order_day']

    total_orders = df.groupby('user_id', as_index=False)['timestamp'].count().rename(columns={'timestamp': 'user_total_orders'})
    user_brand = user_brand.merge(total_orders, on='user_id', how='left').fillna(0)
    user_brand['user_brand_total_share'] = user_brand['user_brand_cnt_orders'] / user_brand['user_total_orders']
    user_brand.drop(columns=['user_total_orders', 'first_order_day', 'last_order_day'], inplace=True)

    user_brand['user_brand_rank'] = user_brand.groupby('user_id')['user_brand_total_share'].rank(
        method='min',
        ascending=False
    )

    return user_brand

def collect_brand_features(df, dp, start_day=1082, end_day=1268):
    reviews = collect_reviews(dp, start_day, end_day)
    reviews_aggr = reviews.groupby(['brand_id'], as_index=False).agg(
        brand_cnt_rating = ('rating', 'count'),
        brand_avg_rating = ('rating', 'mean'),
        brand_min_rating = ('rating', 'min'),
        brand_max_rating = ('rating', 'max'),
    )

    brands = pd.read_parquet('t_ecd_small_partial/dataset/small/brands.pq', engine='fastparquet')
    # fill nan's
    available_embeddings = brands['embedding'].dropna().tolist()
    mean_brand_embedding = np.mean(np.stack(available_embeddings), axis=0)
    brands['embedding'] = brands['embedding'].apply(
        lambda x: x if x is not None else mean_brand_embedding.tolist()
    )
    # aggregate embeddings
    brands = brands.groupby('brand_id')['embedding'].apply(
        lambda x: np.mean(np.stack(x.tolist()), axis=0).tolist() if len(x) > 0 else None
    ).reset_index()

    brands.columns = ['brand_id', 'brand_embedding']

    brands = reviews_aggr.merge(brands, on='brand_id', how='left')
    brands.rename(columns={'brand_id': 'item_brand_id'}, inplace=True)
    return brands

def collect_user_category_features(df, dp, start_day=1082, end_day=1268):
    
    uc_aggr = df.groupby(['user_id', 'item_category'], as_index=False).agg(
        user_category_orders=('timestamp', 'count'),
        user_category_avg_price = ('item_price', 'mean'),
        user_category_min_price = ('item_price', 'min'),
        user_category_max_price = ('item_price', 'max'),
        # user_category_std_price = ('item_price', 'std'),user_category
        user_category_first_order_day = ('day', 'min'),
        user_category_last_order_day = ('day', 'max'),
    )
    uc_aggr['user_category_days_since_first_order'] = end_day - uc_aggr['user_category_first_order_day']
    uc_aggr['user_category_days_since_last_order'] = end_day - uc_aggr['user_category_last_order_day']
    uc_aggr['user_category_lifetime'] = uc_aggr['user_category_last_order_day'] - uc_aggr['user_category_first_order_day']

    total_orders = df.groupby('user_id', as_index=False)['timestamp'].count().rename(columns={'timestamp': 'user_total_orders'})
    uc_aggr = uc_aggr.merge(total_orders, on='user_id', how='left')
    uc_aggr['user_category_orders_share'] = uc_aggr['user_category_orders'] / uc_aggr['user_total_orders']
    uc_aggr.drop(columns=['user_total_orders'], inplace=True)

    return uc_aggr


def collect_user_item_features(df, dp, start_day=1082, end_day=1268):
    events = collect_events(start_day, end_day)
    user_item = aggr_events(events, dp, keys=['user_id', 'item_id'])

    top_subdomain = df.groupby(['user_id', 'item_id'], as_index=False).agg(
        user_item_top_subdomain=('subdomain', lambda x: x.value_counts().index[0]),
        first_purchase_day=('day', 'min'),
        last_purchase_day=('day', 'max')
    )
    top_subdomain['user_item_lifetime'] = top_subdomain['last_purchase_day'] - top_subdomain['first_purchase_day']
    top_subdomain['user_item_days_since_first_purchase'] = end_day - top_subdomain['first_purchase_day']
    top_subdomain['user_item_days_since_last_purchase'] = end_day - top_subdomain['last_purchase_day']
    top_subdomain.drop(columns=['first_purchase_day', 'last_purchase_day'], inplace=True)
    user_item = top_subdomain.merge(user_item, on=['user_id', 'item_id'], how='left')
    user_item.rename(columns={
        'item_added_to_cart_total': 'user_item_added_to_cart_total',
        'item_clicked_total': 'user_item_clicked_total',
        'item_viewed_total': 'user_item_viewed_total'
    }, inplace=True)
    return user_item



def collect_all_features(df, dp, start_day=1082, end_day=1268):
    user_features = collect_user_features(df, dp, start_day, end_day)
    print(f'user_features: {user_features.shape}, nans: {user_features.isna().sum().sum()}')
    
    item_features = collect_item_features(df, dp, start_day, end_day)
    print(f'item_features: {item_features.shape}, nans: {item_features.isna().sum().sum()}')

    brand_features = collect_brand_features(df, dp, start_day, end_day)
    print(f'brand_features: {brand_features.shape}, nans: {brand_features.isna().sum().sum()}')

    user_item_features = collect_user_item_features(df, dp, start_day, end_day)
    print(f'user_item_features: {user_item_features.shape}, nans: {user_item_features.isna().sum().sum()}')

    user_brand_features = collect_user_brand_features(df, dp, start_day, end_day)
    print(f'user_brand_features: {user_brand_features.shape}, nans: {user_brand_features.isna().sum().sum()}')

    user_category_features = collect_user_category_features(df, dp, start_day, end_day)
    print(f'user_category_features: {user_category_features.shape}, nans: {user_category_features.isna().sum().sum()}')

    result_df = df[['user_id', 'item_id', 'item_brand_id']].copy()
    result_df = result_df.merge(user_features, on='user_id', how='left')
    result_df = result_df.merge(item_features, on='item_id', how='left')
    result_df = result_df.merge(brand_features, on='item_brand_id', how='left')
    result_df = result_df.merge(user_item_features, on=['user_id', 'item_id'], how='left')
    result_df = result_df.merge(user_brand_features, on=['user_id', 'item_brand_id'], how='left')
    result_df = result_df.merge(user_category_features, on=['user_id', 'item_category'], how='left')
    
    print('done! nans: ', result_df.isna().sum().sum())
    return result_df


def add_features_to_samples(
    samples_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dp,
    start_day: int = 1082,
    end_day: int = 1268
) -> pd.DataFrame:

    items_mapping = pd.read_parquet('t_ecd_small_partial/dataset/small/retail/items.pq', engine='fastparquet')
    items_mapping['item_id'] = items_mapping['item_id'].apply(lambda x: dp.item_to_idx[x] if x in dp.item_to_idx else None)
    items_mapping = items_mapping.dropna(subset=['item_id'])
    items_mapping['item_id'] = items_mapping['item_id'].astype(np.int64)
    items_mapping = items_mapping.rename(columns={
        'brand_id': 'item_brand_id'
    })

    samples_with_brand = samples_df.merge(items_mapping, on='item_id', how='left')
    
    user_features = collect_user_features(train_df, dp, start_day, end_day)
    print(f'user_features: {user_features.shape}, nans: {user_features.isna().sum().sum()}')

    item_features = collect_item_features(train_df, dp, start_day, end_day)
    print(f'item_features: {item_features.shape}, nans: {item_features.isna().sum().sum()}')
    
    brand_features = collect_brand_features(train_df, dp, start_day, end_day)
    print(f'brand_features: {brand_features.shape}, nans: {brand_features.isna().sum().sum()}')
    
    user_item_features = collect_user_item_features(train_df, dp, start_day, end_day)
    print(f'user_item_features: {user_item_features.shape}, nans: {user_item_features.isna().sum().sum()}')
    
    user_brand_features = collect_user_brand_features(train_df, dp, start_day, end_day)
    print(f'user_brand_features: {user_brand_features.shape}, nans: {user_brand_features.isna().sum().sum()}')
    
    user_category_features = collect_user_category_features(train_df, dp, start_day, end_day)
    print(f'user_category_features: {user_category_features.shape}, nans: {user_category_features.isna().sum().sum()}')
    
    result = samples_with_brand[['user_id', 'item_id', 'item_brand_id']].copy()
    
    result = result.merge(user_features, on='user_id', how='left')
    result = result.merge(item_features, on='item_id', how='left')
    result = result.merge(brand_features, on='item_brand_id', how='left')
    result = result.merge(user_item_features, on=['user_id', 'item_id'], how='left')
    result = result.merge(user_brand_features, on=['user_id', 'item_brand_id'], how='left')
    result = result.merge(user_category_features, on=['user_id', 'item_category'], how='left')
    
    interaction_cols = [
        'user_item_added_to_cart_total', 'user_item_clicked_total', 'user_item_viewed_total',
        'user_item_days_since_first_purchase', 'user_item_days_since_last_purchase',
        'user_item_lifetime',
        'user_brand_days_since_first_order', 'user_brand_days_since_last_order',
        'user_category_days_since_first_order', 'user_category_days_since_last_order',
    ]
    
    for col in interaction_cols:
        if col in result.columns:
            if 'days_since' in col:
                result[col] = result[col].fillna(9999)  # Большое число = "никогда"
            else:
                result[col] = result[col].fillna(0)  # Нет взаимодействий
    
    
    print('done! nans: ', result.isna().sum().sum())
    return result
