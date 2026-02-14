from typing import List, Optional, Dict, Tuple
import numpy as np

class TopPopular:
    def __init__(self):
        self.trained = False

    def fit(self, df, col='train_interactions'):
        counts = {}
        for _, row in df.iterrows():
            for item, _, _ in row[col]:
                if item in counts:
                    counts[item] += 1
                else:
                    counts[item] = 1
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.recommendations = [x[0] for x in counts]
        self.trained = True

    def predict(self, df, topn=10, return_scores=False) -> List:
        assert self.trained, 'Model not fitted!'
        
        items = self.recommendations[:topn]
        
        if return_scores:
            items_with_scores = [(item, 1.0 / (i + 1)) for i, item in enumerate(items)]
            return [items_with_scores] * len(df)
        else:
            return [items] * len(df)