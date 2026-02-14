from typing import List, Optional, Tuple
import numpy as np

class TopPersonal:
    def __init__(self):
        self.trained = False

    def fit(self, df, col='train_interactions'):
        self.user_recommendations = {}
        for idx, row in df.iterrows():
            user_items = {}
            for item, _, _ in row[col]:
                if item in user_items:
                    user_items[item] += 1
                else:
                    user_items[item] = 1
            user_items = sorted(user_items.items(), key=lambda x: x[1], reverse=True)
            self.user_recommendations[idx] = [x[0] for x in user_items]
        self.trained = True

    def predict(self, df, topn=10, return_scores=False) -> List:
        assert self.trained, 'Model not fitted!'
        predictions = []
        for idx, row in df.iterrows():
            if idx in self.user_recommendations:
                items = self.user_recommendations[idx][:topn]
                if return_scores:
                    items_with_scores = [(item, 1.0 / (i + 1)) for i, item in enumerate(items)]
                    predictions.append(items_with_scores)
                else:
                    predictions.append(items)
            else:
                predictions.append([])
        return predictions
