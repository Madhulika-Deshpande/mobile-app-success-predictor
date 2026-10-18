# mobile-app-success-predictor
Machine learning project to predict mobile app install success using Google Play Store data.
Predicts whether a Google Play app will reach a Low, Medium or High install tier from information known before launch (category, size, price, content rating, update recency and more).

Team: Madhulika, Sharvari, Aditi
Course: Machine Learning Lab Project 
Live app: [add deployment link after testing]

1. Project summary
Goal: classify apps into success_tier (Low / Medium / High) using Google Play Store metadata.
Task type: multi-class classification.
Models compared: Logistic Regression, KNN, Decision Tree, SVM, Random Forest, AdaBoost.
Metrics: accuracy, precision, recall, macro-F1, weighted-F1 and confusion matrix.
Note: installs are given only as ranges (e.g. 10,000+), so tiers are approximate.
2. Dataset
Name: Google Play Store Apps (googleplaystore.csv)
Source: https://www.kaggle.com/datasets/lava18/google-play-store-apps
Uploader / author: [confirm on the Kaggle page]
License: [copy from the Kaggle page]
Date accessed: [add date]
Size: 10,841 rows, 13 columns (about 9,659 unique apps after cleaning).
Snapshot: Last Updated dates run from 2010 to August 2018, so this is a 2018 snapshot.

The raw CSV is not modified. To download it yourself:

bash
kaggle datasets download lava18/google-play-store-apps

(If the license allows redistribution, the raw file is also in data/.)

3. Project decisions (agreed at Checkpoint 1)
Item	Decision
Target	success_tier
Low	installs < 10,000
Medium	10,000 to < 1,000,000
High	installs >= 1,000,000
Install value used	lower bound of each range (e.g. 10,000+ -> 10,000)
Duplicate rule	one row per app; keep the row with the highest Reviews
Corrupted row	row 10472 (Category = 1.9) removed
Random seed	RANDOM_STATE = 42 everywhere
Split	80:20 train/test, stratified on success_tier, same split for every model
Preprocessing	scikit-learn Pipeline; imputation, encoding and scaling fitted on training data only
Test set	untouched until final evaluation; never used for tuning
Features

Inputs: category, genres (primary genre), size_mb, size_varies, type, price, content_rating, android_ver_min, days_since_update

Excluded: Rating, Reviews, Installs, App, Current Ver

Rating and Reviews are excluded because they are not known before launch (Reviews also correlates strongly with installs, which would leak the target).
days_since_update = 8 Aug 2018 (latest date in the data) minus Last Updated.
Rating and Reviews are still cleaned and used in EDA to show the leakage argument.
4. Repository structure
data/         raw and cleaned data (or download instructions)
notebooks/    Colab/Jupyter notebooks (cleaning, EDA, models)
src/          cleaning functions, config, prediction function
app/          user interface
models/       saved trained pipeline
figures/      PNG plots (each noted with the notebook that created it)
README.md
requirements.txt
5. Setup
bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt

Library versions are pinned in requirements.txt. The saved model must be loaded with the same scikit-learn version used for training.

6. How to run
Cleaning: run notebooks/01_cleaning.ipynb (or the cleaning script in src/). It produces the cleaned file in data/.
EDA: run notebooks/02_eda.ipynb. Figures are saved to figures/.
Models: run notebooks/03_models.ipynb. It trains and evaluates all models on the same split and saves the best pipeline to models/.
App:
bash
   # Streamlit
   streamlit run app/app.py
   # or Gradio
   python app/app.py
7. Results

Fill this in only with numbers from models we actually ran.

Model	Accuracy	Macro-F1	Weighted-F1	Precision (macro)	Recall (macro)
Logistic Regression					
KNN					
Decision Tree					
SVM					
Random Forest					
AdaBoost					

Best model: [name]. Confusion matrices and feature importance are in figures/.

8. Limitations
Install tiers are approximate because Google Play reports installs as ranges.
The data is a 2018 snapshot, so patterns may differ from the current store.
Size, update date and Android version describe an app after release, so they are only partly available before launch.
Feature importance shows what the model uses, not what causes success. We make no causal claims.
Predictions are estimates for guidance, not guarantees.
9. Team contributions
Phase	Madhulika	Sharvari	Aditi
Dataset & setup	GitHub repo	Source, license, citation	Inspect CSV, confirm target
Cleaning	Installs, Price, Type	Reviews, Size, Rating	Category/Genres, Last Updated, Android Ver, Content Rating, duplicates
EDA	Plots for her columns	Plots for her columns	Plots for her columns, class balance
Models	Logistic Regression, KNN	Decision Tree, SVM	Random Forest, AdaBoost
Evaluation	Her 2 models	Her 2 models	Her 2 models, feature importance
UI	Input form with validation	Model loading and prediction	Results display, testing, deployment
Report	Abstract, Introduction, Literature Review	Methodology, Preprocessing, References	Results & Discussion, UI & Deployment, README
10. References

[Add the dataset citation and the 5+ papers here.]

Dataset: [Uploader]. Google Play Store Apps. Kaggle. Accessed [date]. https://www.kaggle.com/datasets/lava18/google-play-store-apps
