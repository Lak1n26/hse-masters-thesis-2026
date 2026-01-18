import numpy as np
from tqdm import tqdm


class iALS:
    """
    Collaborative Filtering for Implicit Feedback Datasets
    Paper: http://yifanhu.net/PUB/cf.pdf
    """

    def __init__(self, n_factors: int = 100, alpha: float = 1.0, reg_coef = 0.01, idx_to_item: dict = None):
        self.n_factors = n_factors
        self.alpha = alpha
        self.reg_coef = reg_coef
        self.idx_to_item = idx_to_item
        self.user_factors = None  # матрица X
        self.item_factors = None  # матрица Y

    def fit(self, interactions: np.ndarray, n_iterations: int=10):
        self.interactions = interactions
        n_users, n_items = self.interactions.shape
        self.user_factors = np.random.normal(size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(size=(n_items, self.n_factors))

        lambda_eye = self.reg_coef * np.eye(self.n_factors)
        
        for iter in range(n_iterations):
            print(f'Iter № {iter + 1}/{n_iterations}:')
            
            YtY = self.item_factors.T @ self.item_factors
            for u in tqdm(range(n_users), desc='Updating users'):
                indices = self.interactions[u].nonzero()[0]
                if len(indices) == 0:
                    continue
                
                c_u = 1.0 + self.alpha * self.interactions[u, indices]
                Y_u = self.item_factors[indices]
                YtCY = Y_u.T @ (c_u[:, None] * Y_u)
                YtCp = Y_u.T @ c_u  # p_u = 1 for interacted items
                
                # Solve: (YtY + YtCY - Y_u^T @ Y_u + λI) x_u = YtCp
                A = YtY + YtCY - (Y_u.T @ Y_u) + lambda_eye
                self.user_factors[u] = np.linalg.solve(A, YtCp)
            
            XtX = self.user_factors.T @ self.user_factors
            for i in tqdm(range(n_items), desc='Updating items'):
                indices = self.interactions[:, i].nonzero()[0]
                if len(indices) == 0:
                    continue
                c_i = 1.0 + self.alpha * self.interactions[indices, i]
                X_i = self.user_factors[indices]
                XtCX = X_i.T @ (c_i[:, None] * X_i)
                XtCp = X_i.T @ c_i  # p_i = 1 for users who interacted
                
                # Solve: (XtX + XtCX - X_i^T @ X_i + λI) y_i = XtCp
                A = XtX + XtCX - (X_i.T @ X_i) + lambda_eye
                self.item_factors[i] = np.linalg.solve(A, XtCp)

    def predict(self, df, topn: int = 100):
        # возвращает top-k айтемов для каждого юзера (айтемы с которыми юзер взаимодействовал не должны попасть в рекомендации)
        scores = self.user_factors @ self.item_factors.T
        predictions = []
        
        for row_idx, (df_idx, _) in enumerate(df.iterrows()):
            user_scores = scores[row_idx].copy()
            
            # проставим -inf тем айтемам, с которыми юзер уже взаимодействовал
            interacted_items = self.interactions[row_idx].nonzero()[0]
            user_scores[interacted_items] = -np.inf
            
            if topn >= len(user_scores):
                top_indices = np.argsort(user_scores)[::-1]
            else:
                top_indices = np.argpartition(user_scores, -topn)[-topn:]
                top_indices = top_indices[np.argsort(user_scores[top_indices])[::-1]]
            
            if self.idx_to_item is not None:
                top_items = [self.idx_to_item[idx] for idx in top_indices[:topn]]
            else:
                top_items = top_indices[:topn].tolist()
            
            predictions.append(top_items)
        
        return predictions