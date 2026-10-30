# Pharmacy Product Search & Recommendation (europharma.kz)

End-to-end text ML project: web scraping of a Kazakhstan online pharmacy, data cleaning, EDA,
and a TF-IDF–based search and "similar products" recommendation system.

## Pipeline

| Notebook | What it does |
|---|---|
| `01_parsing.ipynb` | Scrapes europharma.kz with `requests` + `BeautifulSoup`: category links → product links (**6,252 unique products**) → product cards (name, price, availability, country, manufacturer, description, categories, pharmacies). |
| `02_cleaning.ipynb` | Removes items without price/description, parses list-like fields, cleans descriptions and categories → **3,417 products** in `processed_products_data.csv`. |
| `03_eda.ipynb` | Price distribution (with and without outliers), top categories, top pharmacies by assortment, manufacturer countries. |
| `04_search_and_recommendation.ipynb` | Text preprocessing (lowercasing, Russian stemming, stop words) → TF-IDF (5,000 features) → clustering (KMeans on TF-IDF / SVD / PCA) and KNN with cosine distance for search and recommendations. |

## Results

**Clustering (Silhouette Score):**

| Representation | Silhouette |
|---|---|
| TF-IDF + KMeans (k=8) | 0.045 |
| TF-IDF → SVD(100) + KMeans (k=8) | 0.11 |
| TF-IDF → PCA(50) + KMeans (k=6) | 0.132 |

Product texts do not form well-separated clusters; dimensionality reduction helps but the structure stays weak.
The main value of the project is search and recommendation, not segmentation.

**Search and recommendations:** the final system (TF-IDF → SVD → KNN, cosine) takes a free-text query
and returns the closest products. Example: for `Декарис 50 мг` the nearest neighbours are other anthelmintics
(Декарис 150 мг, Пирантел, Гельминтокс).

## Limitations & next steps

- TF-IDF matches **words, not meaning**: the query «от горла» returns only mouth rinses, because they share
  vocabulary with the query, while throat lozenges and sprays with different wording are missed.
- Mean cosine similarity of neighbours is reported in the notebook, but it is not comparable across
  representations of different dimensionality and is not a measure of relevance. A proper evaluation
  would need a labelled set of queries (precision@k).
- Next step: replace TF-IDF with multilingual sentence embeddings, store them in a vector index,
  and add an LLM on top to answer questions over the retrieved products (RAG).

## Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order. Data files (`products_data.csv`, `processed_products_data.csv`) are not stored
in the repository; `01_parsing.ipynb` recreates them. Scraped data reflects the site at collection time (December 2025).

## Stack

Python, pandas, requests, BeautifulSoup, NLTK, scikit-learn (TF-IDF, TruncatedSVD, PCA, KMeans, NearestNeighbors),
matplotlib, seaborn, wordcloud.
