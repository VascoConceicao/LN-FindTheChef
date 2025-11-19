# Find the Chef – ML Project

## Overview
This project predicts the **chef** who created a recipe based on its text fields (name, ingredients, steps, etc.).  
It follows the three-phase workflow provided in the assignment:

1. **Setup, Data Exploration, and Baselines**  
2. **Advanced Modeling and Iteration**  
3. **Finalization and Submission**

## Project Setup

### 1. Virtual Environment

Create a Python virtual environment:

```bash
python -m venv env
```

Activate it:

- **Windows PowerShell**

```bash
.\env\Scripts\Activate
```

- **macOS/Linux**

```bash
source env/bin/activate
```

### 2. Install Dependencies

Install essential libraries:

```bash
pip install requirements
```

Or manually:

```bash
pip install pandas scikit-learn nltk numpy matplotlib seaborn jupyter lightgbm
```

### 3. Copy Data

Copy the data files into the `/data` folder.

## Testing

The `test.ipynb` tests if your setup is correct and you have all the necessary dependencies installed.

## Data Preprocessing and Analysis

The `analysis_and_preprocessing.ipynb` script preprocesses the `data/train.csv` file to prepare it for machine learning.

It performs the following steps:
- **Combines** text from `recipe_name`, `description`, `tags`, `steps`, and `ingredients` into a single document per recipe.
- **Cleans** the text by making it lowercase and removing punctuation and numbers.
- **Removes** common English stopwords.

The final, model-ready dataset is saved as `data/preprocessed_data.csv`.

## To-do List

✅ **Project Setup**

- Set up Python 3 programming environment. Use a virtual environment (e.g., venv) to manage dependencies.
- Install essential libraries.

✅ **Data Exploration and Preprocessing Strategy**

- **Load Data**: Load the `train.csv` file using `pandas`.
- **Initial Analysis** (Look at the data!):
  - Check the class distribution: How many recipes does each of the 6 chefs have? Is the dataset unbalanced?
  - Examine the fields: `chef_id`, `recipe_name`, `data`, `tags`, `steps`, `description`, `ingredients`, `n_ingredients`.
  - Look at a few examples for each chef. Do they use different vocabulary? Are their tags or description styles different?
  - Note how list-like fields (`tags`, `steps`, `ingredients`) are stored.

- **Plan Preprocessing**:
  - Decide how we will clean the text data (e.g., convert to lowercase, remove punctuation).
  - Strategize on how to combine the different text fields (`recipe_name`, `tags`, `steps`, `description`, `ingredients`) into a single input for your model.
  - Be careful about blindly removing stop words.

✅ **Create a Validation Set**

- Before training, split our `train.csv` data into a training set and a validation set (e.g., 80% train, 20% validation).

- We must report results on our own test set, and this validation set will serve that purpose. All performance metrics we generate before the final submission will be on this set.

✅ **Build and Evaluate Baseline Models**

- **Weak Baseline** (Target: >30.0% Accuracy for 2 points):
  1. Apply our text preprocessing to the `description` field of our new training split.
  2. Use `TfidfVectorizer` from `scikit-learn` to convert the text into numerical features.
  3. Train a `Support Vector Classifier` (`SVC`) model on these features and the chef IDs.
  4. Evaluate the model's accuracy on our validation set.

- **Strong Baseline** (Target: >43.0% Accuracy for 2 more points):
  1. Combine all relevant text fields into a single text block for each recipe.
  2. Apply our text preprocessing to this combined text.
  3. Use `TfidfVectorizer` to create features from this combined text.
  4. Train a new `SVC` model.
  5. Evaluate its accuracy on our validation set.

✅ **Feature Engineering**

- **Text Features**: Experiment with `TfidfVectorizer` parameters (e.g., `ngram_range`, `min_df`, `max_df`).
- **Numerical Features**: Use the `n_ingredients` column. We could also create new features like the number of steps, length of the description, etc. We may need to combine these with the TF-IDF features.
- **Categorical Features**: Process the `tags` field. Can we use these tags as features?

✅ **Experiment with Different Models**

- Go beyond `SVC`. Try other classification algorithms suitable for text data:
  - `Multinomial Naive Bayes`
  - `Logistic Regression`
  - `Random Forest` or `Gradient Boosting` models (like XGBoost, LightGBM).
- Systematically train and evaluate each model on our validation set. Keep a log of our experiments and their results for the paper. Use tools like `scikit-learn`'s `Pipeline` to ensure our preprocessing is applied consistently.

🟦 **Error Analysis**

- For our best-performing model:
  - Generate a confusion matrix. Which chefs are most often confused with each other?
  - Inspect wrong predictions. Look at specific recipes our model got wrong. Why do we think it failed? Was the recipe text ambiguous? Is the label questionable? 
  - These observations will form the core of the "discussion" section in our paper.

🟦 **Write the Short Paper**

- **Structure**: Use the provided template. The paper is a maximum of 2 pages.
  - **Introduction**: Briefly describe the task: mapping a recipe to its creator chef.
  - **Methodology**: Describe our data preprocessing, feature engineering, and the different models we tested (including the baselines).
  - **Results & Discussion**: Present the performance (accuracy) of our models on our validation set in a clear table. Include our error analysis and confusion matrix. Discuss limitations of the data or our system.
  - **Conclusion**: Summarize our work and key findings.
- **Citations**: Identify all external sources/code we used.

🟦 **Generate Final Test Predictions**

1. **Select Best Model**: Choose our single best model based on our validation set performance.
2. **Retrain**: Retrain this model on the entire `train.csv` dataset.
3. **Predict**:
   - Load the `test-no-labels.csv` file.
   - Apply the exact same preprocessing and feature extraction steps we used for our training data.
   - Predict the `chef_id` for each recipe.
   - Save the predictions to a file named `results.txt`, with one `chef_id` per line, ensuring the line numbers match the test file.

🟦 **Final Submission**

- **Package**: Create a zip file named `NUM.zip` (where NUM is our group number).
- **Contents**: The zip file must contain:
  1. Our short paper: `NUM.pdf`.
  2. Our code: `.py` files or `.ipynb` notebooks.
  3. Our final predictions: `results.txt`.
- **Submit**: Upload the zip file to Fenix before the deadline. Double-check everything before submitting.

### Repository Structure

```
find-the-chef/
│
├── data/                               # Place for train/test CSVs
├── env/                                # Virtual environment (ignored by git)
├── preprocessing_and_splitting.ipynb   # Initial analysis, preprocessing of data and splitting
├── baseline_models.ipynb               # Building and evaluating baseline models
├── feature_engineering.ipynb           # Experimenting with text, numerical and categorical features
├── experiment_diff_models.ipynb        # Experimenting with Logistic Regression, LightGBM and Multinomial Naive Bayes
├── requirements.txt                    # List of dependencies
└── test.ipynb                          # Test notebook
```

