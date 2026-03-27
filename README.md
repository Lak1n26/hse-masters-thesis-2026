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
6. ARGUS (by Yandex) - [argus.ipynb](notebooks/argus.ipynb)

Промежуточная презентация - [presentation.pdf](presentation.pdf)

---

Basket-специфичные модели:
- ReCANet ([статья](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/ariannezhad-2022-recanet.pdf)) - [recanet.ipynb](notebooks/recanet.ipynb) - модель рекомендаций следующей корзины
- NPA ([статья](https://arxiv.org/pdf/2401.16433)) - [npa.ipynb](notebooks/npa.ipynb) - модель рекомендаций внутри корзины
- Beacon ([статья](https://www.ijcai.org/proceedings/2019/0389.pdf)) - [beacon.ipynb](notebooks/beacon.ipynb) - модель рекомендаций следующей корзины
- SAFEREc (by T-Bank) ([статья](https://arxiv.org/pdf/2412.14302)) - [saferec.ipynb](notebooks/saferec.ipynb) - модель рекомендаций следующей корзины

В работе:
1. Реализация собственной архитектуры, ее бенчмаркинг, две ключевые идеи:
- Embedding-based Retrieval (первая попытка в [embedding_retrieval.ipynb](notebooks/embedding_retrieval.ipynb))
- Generative Retrieval (реализую идею из https://arxiv.org/abs/2305.05065, первая попытка в [rqvae.ipynb](notebooks/rqvae.ipynb))
