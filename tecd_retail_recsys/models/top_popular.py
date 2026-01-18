from typing import List, Optional
import numpy as np

class TopPopular:
    def __init__(self):
        self.trained = False

    def fit(self, df, col='train_interactions'):
        counts = {}
        for _, row in df.iterrows():
            for item, _ in row[col]:
                if item in counts:
                    counts[item] += 1
                else:
                    counts[item] = 1
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.recommendations = [x[0] for x in counts]
        self.trained = True

    def predict(self, df, topn=10)  -> List[np.ndarray]:
        assert self.trained, 'Model not fitted!'
        return [self.recommendations[:topn]]*len(df)