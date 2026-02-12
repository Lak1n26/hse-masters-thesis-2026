import numpy as np
import pandas as pd

def calculate_avg_prices(grouped_data):
    """
    Вычисляет среднюю цену товаров для каждого пользователя
    по train, val и test взаимодействиям.
    """
    # Функция для вычисления средней цены из списка кортежей (item_id, timestamp, price)
    def avg_price(interactions):
        if len(interactions) == 0:
            return np.nan
        prices = [price for _, _, price in interactions]
        return np.mean(prices)
    
    # Вычисляем средние цены для каждого типа взаимодействий
    grouped_data['avg_train_price'] = grouped_data['train_interactions'].apply(avg_price)
    grouped_data['avg_val_price'] = grouped_data['val_interactions'].apply(avg_price)
    grouped_data['avg_test_price'] = grouped_data['test_interactions'].apply(avg_price)
    
    return grouped_data


def calculate_overall_avg_price(grouped_data, interaction_col='train_interactions'):
    """
    Вычисляет общую среднюю цену товаров по всем пользователям.
    """
    total_price = 0
    total_count = 0
    
    for interactions in grouped_data[interaction_col]:
        for item_id, timestamp, price in interactions:
            total_price += price
            total_count += 1
    
    return total_price / total_count if total_count > 0 else np.nan


def get_avg_recs_price(joined, item_to_price,col='toppopular_recs'):
    for interactions in joined[col]:
        total_price = 0
        total_cnt = 0
        for item_id in interactions:
            total_price += item_to_price[item_id]
            total_cnt += 1
    return total_price / total_cnt


def get_item_to_price(dp, path='t_ecd_small_partial/dataset/small/retail/items.pq'):
    item_to_price = pd.read_parquet(path)
    item_to_price['price'] = item_to_price['price'].fillna(0)
    item_to_price = item_to_price[item_to_price['item_id'].isin(dp.item_to_idx.keys())]
    item_to_price = dict(zip(item_to_price['item_id'], item_to_price['price']))
    item_to_price = {dp.item_to_idx[id]: price for id, price in item_to_price.items()}
    return item_to_price