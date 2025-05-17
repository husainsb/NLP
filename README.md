# Nationality Classification using Embeddings & ML

We have ~7700 names with nationalities. Aim is to predict/classify the nationality from given names.

* First, we will convert names to embeddings using OpenAI Text embeddings-3 small model. This will produce 1536 dimensions for each given name
* These embeddings will then be stored in vector DB like FAISS
* We will apply PCA on these mbeddings to reduce dimensions from 1536 to 20. This will help in visualization and running ML
* We will try Random Forest & XGBoost classification with hyperparamter tuning to find best classification model

*Note: Due to dual nationality of few people, the missclassification of Nationalities can occur. For e.g. a person with name like an Arab might have a reported nationality of Western country.*
