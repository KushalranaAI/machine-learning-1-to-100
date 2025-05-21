# Data Cleaning: Key Topics with Detailed Examples

Data cleaning is a vital step in preparing raw data for analysis or machine learning. It ensures the dataset is accurate, consistent, and free of errors that could skew results. Let's dive into each topic with detailed explanations and examples.

---

### 1. Data Type Correction

**What it is:** Data type correction involves ensuring that each column in a dataset has the appropriate data type (e.g., integer, float, string, datetime) for analysis.

**Why it matters:** Incorrect data types can cause errors in calculations, visualizations, or modeling. For instance, if numbers are stored as strings, you can't perform arithmetic operations until they're converted.

**How to do it:**
- Use Pandas' `astype()` to convert columns to the desired type.
- Use `pd.to_datetime()` for date strings.

**Example:**
Suppose you have a dataset where numerical values and dates are stored as strings:

```python
import pandas as pd

# Sample DataFrame with incorrect data types
df = pd.DataFrame({
    'id': ['1', '2', '3'],           # Should be integer
    'price': ['10.5', '20.0', '15.3'], # Should be float
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'] # Should be datetime
})

# Correct the data types
df['id'] = df['id'].astype(int)
df['price'] = df['price'].astype(float)
df['date'] = pd.to_datetime(df['date'])

# Check the result
print(df)
print("\nData types:")
print(df.dtypes)
```

**Output:**
```
   id  price       date
0   1   10.5 2023-01-01
1   2   20.0 2023-01-02
2   3   15.3 2023-01-03

Data types:
id                int64
price           float64
date     datetime64[ns]
```

**When to use:** Apply data type correction early in the cleaning process to ensure downstream operations (e.g., calculations or plotting) work correctly.

---

### 2. Handling Missing Values

**What it is:** Missing values (e.g., `NaN`, `None`) occur when data is incomplete. You can either remove them or fill them with appropriate values.

**Why it matters:** Missing data can distort statistics (e.g., means) or cause errors in models if not handled properly.

**Strategies and When to Use:**
- **Removing rows with missing values:** Best when the dataset is large and missing data is minimal/very few in number (e.g., <5% of data).
- **Filling with mean:** Use for numerical data that’s normally distributed, as it preserves the average.
- **Filling with median:** Use for numerical data with outliers or skewed distributions, as it’s more robust.
- **Filling with mode:** Use for categorical data, as it reflects the most common category.

**Examples:**

- **Removing missing values:**
```python
df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})
df = df.dropna()
print(df)
```
**Output:**
```
     A  B
0  1.0  4
2  3.0  6
```

- **Filling with mean (numerical data):**
```python
df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})
df['A'] = df['A'].fillna(df['A'].mean())  # Mean of [1, 3] is 2
print(df)
```
**Output:**
```
     A  B
0  1.0  4
1  2.0  5
2  3.0  6
```

- **Filling with median (numerical data with outliers):**
```python
df = pd.DataFrame({'A': [1, None, 100, 2]})
df['A'] = df['A'].fillna(df['A'].median())  # Median of [1, 2, 100] is 2
print(df)
```
**Output:**
```
       A
0    1.0
1    2.0
2  100.0
3    2.0
```

- **Filling with mode (categorical data):**
```python
df = pd.DataFrame({'category': ['A', None, 'B', 'A']})
df['category'] = df['category'].fillna(df['category'].mode()[0])  # Mode is 'A'
print(df)
```
**Output:**
```
  category
0        A
1        A
2        B
3        A
```

**When to use:**
- **Remove:** When missing data is rare and won’t impact overall trends.
- **Mean:** When data is numerical and evenly distributed.
- **Median:** When data is numerical and has outliers or skewness.
- **Mode:** When data is categorical.

---

### 3. Handling Inconsistent Categories

**What it is:** Categorical variables often have inconsistencies like misspellings (e.g., 'Californa' vs. 'California'), different capitalizations (e.g., 'Male' vs. 'male'), or too many unique values.

**Why it matters:** Inconsistent categories can lead to incorrect grouping or analysis, as the same category might be treated as distinct due to minor differences.

**How to handle:**
- Convert text to a consistent case (e.g., lowercase).
- Correct misspellings manually or with mappings.
- Group rare categories into an 'Other' category if there are too many unique values.

**Example:**
```python
# Sample DataFrame with inconsistent categories
df = pd.DataFrame({'gender': ['Male', 'male', 'Female', 'female', 'Femal']})

# Standardize to lowercase and fix misspellings
df['gender'] = df['gender'].str.lower().replace({'femal': 'female'})

print(df)
print("\nValue counts:")
print(df['gender'].value_counts())
```

**Output:**
```
   gender
0    male
1    male
2  female
3  female
4  female

Value counts:
female    3
male      2
```

**When to use:** Standardize categories before analysis or modeling to ensure accurate grouping and consistency.

---

### 4. Standardizing Text Data

**What it is:** Standardizing text data involves making text uniform by removing variations in case, punctuation, or formatting.

**Why it matters:** Inconsistent text (e.g., 'Hello!' vs. 'hello') can complicate analysis, especially for tasks like natural language processing (NLP).

**How to do it:**
- Convert text to lowercase with `str.lower()`.
- Remove punctuation using Python’s `string` module.

**Example:**
```python
import string

# Sample DataFrame with varied text
df = pd.DataFrame({'text': ['Hello, World!', 'PYTHON is Great.', 'Data-Cleaning.']})

# Convert to lowercase and remove punctuation
df['text'] = df['text'].str.lower().apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

print(df)
```

**Output:**
```
             text
0     hello world
1  python is great
2    datacleaning
```

**When to use:** Standardize text data before text analysis or feeding it into machine learning models.

---

### 5. Removing Duplicates

**What it is:** Duplicate rows are identical entries in a dataset that need to be removed.

**Why it matters:** Duplicates can inflate statistics (e.g., averages) or bias model training, leading to incorrect conclusions.

**How to do it:** Use Pandas’ `drop_duplicates()` to remove duplicate rows.

**Example:**
```python
# Sample DataFrame with duplicates
df = pd.DataFrame({
    'id': [1, 2, 2, 3],
    'value': ['a', 'b', 'b', 'c']
})

# Remove duplicates
df = df.drop_duplicates()

print(df)
```

**Output:**
```
   id value
0   1     a
1   2     b
3   3     c
```

**When to use:** Check for and remove duplicates early in the cleaning process to ensure accurate results.

---

### 6. General Errors in Data Type Conversion

**What it is:** Errors occur during data type conversion when the data contains invalid or mixed values.

**Common issues:**
- **ValueError:** Trying to convert a non-numeric string (e.g., 'abc') to a number.
- **Mixed data types:** A column with both numbers and strings.
- **NaN values:** Missing data complicating conversions.

**How to handle:**
- Use `pd.to_numeric()` with `errors='coerce'` to convert invalid values to `NaN`.
- Clean invalid entries beforehand if possible.

**Example:**
```python
# Sample DataFrame with mixed data
df = pd.DataFrame({'A': ['1', '2', 'three', '4.5']})

# Convert to numeric, coercing invalid values to NaN
df['A'] = pd.to_numeric(df['A'], errors='coerce')

print(df)
```

**Output:**
```
     A
0  1.0
1  2.0
2  NaN
3  4.5
```

**When to use:** Use `errors='coerce'` when you suspect invalid values and want to handle them without crashing the process.

---

### Conclusion

Data cleaning ensures your dataset is ready for analysis by addressing:
- **Data Type Correction:** Proper formats for operations.
- **Missing Values:** Removal or imputation based on context.
- **Inconsistent Categories:** Uniform categories for accurate grouping.
- **Text Standardization:** Consistent text for analysis.
- **Duplicates:** Removal for unbiased results.
- **Conversion Errors:** Graceful handling of invalid data.
