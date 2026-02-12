from typing import List, Optional
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

    def predict(self, df, topn=10) -> List[np.ndarray]:
        assert self.trained, 'Model not fitted!'
        predictions = []
        for idx, row in df.iterrows():
            if idx in self.user_recommendations:
                predictions.append(self.user_recommendations[idx][:topn])
            else:
                predictions.append([])
        return predictions
