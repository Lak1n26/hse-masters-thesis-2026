# Нейросетевые подходы к разработке рекомендаций по повышению продаж в товарных корзинах

**Neural Network Approaches to Up-sell Recommendations on Shopping Carts**

> Магистерская диссертация ВШЭ 2026 | Исследовательский проект

---

## О проекте

Репозиторий содержит эксперименты, ноутбуки и модели, посвящённые задаче построения рекомендательных систем для повышения продаж в товарных корзинах.

В проекте представлены:

- базовые подходы;
- современные SOTA-модели;
- basket-специфичные архитектуры;
- ансамбли моделей;
- собственные архитектуры, разработанные в рамках исследования.

---

## Структура экспериментов

### Baseline-модели

- [baseline.ipynb](notebooks/baseline.ipynb)

### SOTA-модели

- [BERT4Rec](notebooks/bert4rec.ipynb)
- [SASRec](notebooks/sasrec.ipynb)
- [BPR](notebooks/bpr.ipynb)
- [BiVAE](notebooks/bivae.ipynb)
- [LightFM](notebooks/lightfm.ipynb)
- [LightGCN](notebooks/lightgcn.ipynb)
- [ARGUS](notebooks/argus.ipynb)

### Ансамблирование моделей

- [CatBoost Ranker](notebooks/catboost.ipynb)
- [Reciprocal Rank Fusion](notebooks/reciprocal_rank_fusion.ipynb)

### Basket-специфичные модели

- [ReCANet](notebooks/recanet.ipynb) — модель рекомендаций следующей корзины, [статья](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/ariannezhad-2022-recanet.pdf)
- [NPA](notebooks/npa.ipynb) — модель рекомендаций внутри корзины, [статья](https://arxiv.org/pdf/2401.16433)
- [Beacon](notebooks/beacon.ipynb) — модель рекомендаций следующей корзины, [статья](https://www.ijcai.org/proceedings/2019/0389.pdf)
- [SAFEREc](notebooks/saferec.ipynb) — модель рекомендаций следующей корзины, [статья](https://arxiv.org/pdf/2412.14302)

### Собственные архитектуры

- [CRUSH — CatBoost Recommender for User-Specific History](notebooks/crush.ipynb)
- [KATUSHA — Knowledge-based Architecture to Transactional User Sequence and Historical Affinity](notebooks/katusha.ipynb)
- [KATUSHA with price-targeting](notebooks/katusha_business_target.ipynb)

---
