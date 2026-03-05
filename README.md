# Нейросетевые подходы к разработке рекомендаций по повышению продаж в товарных корзинах

**Neural Network Approaches to Up-sell Recommendations on Shopping Carts**

> **Магистерская диссертация ВШЭ 2026** | Исследовательский проект

---

1. План работы и обзор литературы - [HSE_Masters_Thesis_Lyapin.pdf](HSE_Masters_Thesis_Lyapin.pdf)
2. Baseline-модели - [baseline.ipynb](notebooks/baseline.ipynb)
3. Эксперименты с SOTA-моделями:
    - [bert4rec.ipynb](notebooks/bert4rec.ipynb)
    - [sasrec.ipynb](notebooks/sasrec.ipynb)
    - [bpr.ipynb](notebooks/bpr.ipynb)
    - [bivae.ipynb](notebooks/bivae.ipynb)
    - [lightfm.ipynb](notebooks/lightfm.ipynb)
    - [lightgcn.ipynb](notebooks/lightgcn.ipynb)
4. CatBoost Ranker - [catboost.ipynb](notebooks/catboost.ipynb)
5. Reciprocal Rank Fusion - [reciprocal_rank_fusion.ipynb](notebooks/reciprocal_rank_fusion.ipynb)

Промежуточная презентация - [presentation.pdf](presentation.pdf)

---

В работе:
1. Реализация собственной архитектуры, ее бенчмаркинг, две ключевые идеи:
- Embedding-based Retrieval (первая попытка в [embedding_retrieval.ipynb](notebooks/embedding_retrieval.ipynb))
- Generative Retrieval (реализую идею из https://arxiv.org/abs/2305.05065, первая попытка в [rqvae.ipynb](notebooks/rqvae.ipynb))

---

В планах:
1. Бенчмаркинг Basket-специфичных моделей (NPA, Beacon, etc)