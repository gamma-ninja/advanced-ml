# BCG Gamma Training session - Syllabus

## Session 1: Machine learning and time series (4H) [slides / notebooks / VSCode for assignment]

- Time series packages and ML problems
- Dealing with datetime columns and index
- Resampling
- Visualization
- Encoding temporal data as features (day of the week, holidays)
- **Hands on assignment:** Complete some Pandas commands on a case study and submit your work in a Pull-Request on GitHub (or equivalent).

## Session 2: Advanced scikit-learn (4H) [notebooks / VSCode for assignment]

- Develop a full case study with a supervised learning task
- scikit-learn API for Pipeline, ColumnTransformer, cross-validation
- Writing your own scikit-learn estimator, transformer, CV splitter.
- Understanding Caching / Parallel execution based on joblib Memory and Parallel
- Understanding the internals of a scikit-learn estimator
- **Hands on assignment:** Writing an estimator and CV splitter with scikit-learn using a CI

## Session 3: Gradient Boosting Decision Trees (GBDT) (4H) [slides / notebooks]
- Gradient-boosting from inside out
    - Theory of GBRT
    - Understanding XGBoost
        - Slides to explain the algorithm
        - What is the computational complexity?
        - Show a pure Python implementation of GBRT implementation
        - Show how to profile with snakeviz
        - Where is the parallelism?
        - Feature binning
- Efficient Tuning a GBRT
    - What are the important parameters?
    - What tools to automate tuning?

## Session 4: Deep Learning “under the hood” (4H) [slides / notebooks / VSCode for assignment]
- Understanding backpropagation hands on:
    - From Numpy to PyTorch for a multi-layer perceptron (MLP)
    - From MLP to CNNs and RNNs (backpropagation through time)
- Understanding stochastic gradient descent

- **Hands on:** Case study on music generation with recurrent neural networks

## Requirements

- `numpy, pandas, scipy, matplotlib`
- `fastparquet`
- `folium`
- `statsmodel`
- `prophet`
- `pytorch`
- `scikit-learn`
- `jupyter`
- `pytest`
