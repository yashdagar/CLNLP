# Experiments Report

This report covers two experiments: an exploratory analysis of an employee dataset using pandas and matplotlib (Experiment 1), and basic text preprocessing for Natural Language Processing (Experiment 2).

## Experiment 1: Employee Dataset Analysis

**Notebook:** `Experiment-1/Experiment-1.ipynb`
**Dataset:** `employee_information_100.csv`, a table of 100 employee records with columns for department, salary, gender, years of experience, and age.

### Objective

Explore the employee dataset and answer ten analytical questions (experiments 1.1 to 1.10) using pandas for aggregation and matplotlib for visualization.

### Method

The notebook first loads the CSV, strips whitespace from column names, and auto-detects the required columns by normalizing names (lowercase, no spaces or underscores) and matching them against common variants, for example `Salary`, `AnnualSalary`, `Income`, or `Pay` for the salary column. If any required column cannot be found, the notebook reports which ones are missing instead of failing mid-analysis.

With the columns resolved, it runs the ten sub-experiments:

| # | Question | Technique |
|---|----------|-----------|
| 1.1 | Average salary per department | `groupby().mean()`, bar chart |
| 1.2 | Employee count per department | `value_counts()` |
| 1.3 | Male and female percentage | `value_counts()` normalized, pie chart |
| 1.4 | Salary distribution | histogram, 10 bins |
| 1.5 | Experience vs salary relationship | scatter plot |
| 1.6 | Top 10 highest-paid employees | `nlargest(10)` |
| 1.7 | Highest salary per department | `groupby().max()` |
| 1.8 | Employees earning above the overall mean | boolean filtering against `mean()` |
| 1.9 | Average experience per department | `groupby().mean()` |
| 1.10 | Age distribution | histogram, 10 bins |

### Results

The notebook produces, for each sub-experiment, a labeled console section plus four charts: a department salary bar chart, a gender pie chart, and histograms of salary and age. The scatter plot in 1.5 visualizes whether salary rises with years of experience, and 1.8 splits the workforce into above-average and below-average earners relative to the overall mean salary. The source CSV lives in the original Colab environment, so the concrete figures are reproduced by re-running the notebook there; they are not restated here.

### Conclusion

The experiment demonstrates the standard pandas exploratory workflow, loading and cleaning data, grouping and aggregating, filtering with boolean masks, and visualizing distributions and relationships. The column auto-detection makes the notebook robust to small differences in dataset headers.

## Experiment 2: Basic Text Preprocessing

**Notebook:** `Experiment-2/Experiment-2.ipynb`
**Inputs:** three small English text files about NLP, `2.1_text_data.txt` (787 characters), `2.2_tokenization_data.txt`, and `2.3_clean_data.txt`.

### Objective

Apply the first steps of an NLP pipeline: cleaning raw text, tokenizing it into sentences and words with three different approaches, and removing stop words.

### 2.1 Text cleaning

The raw text deliberately contains noise. Before cleaning, the notebook counts what will be removed, and on the provided input it finds:

- 32 uppercase letters
- 27 punctuation marks
- 3 numbers (2026, 1000, 12345)
- 7 places with runs of extra whitespace, 9 redundant whitespace characters in total

Cleaning then lowercases the text, strips punctuation and digits with character filters, and collapses whitespace via `' '.join(text.split())`. The result shrinks from 787 to 735 characters and reads as a single normalized line, for example: "natural language processing nlp is a branch of artificial intelligence it helps computers understand and process human language...".

### 2.2 Tokenization, three ways

The same paragraph is tokenized with NLTK (`sent_tokenize`, `word_tokenize`), with spaCy (`en_core_web_sm`, iterating `doc.sents` and non-space tokens), and with plain Python. The plain Python version ends sentences at `.`, `!`, or `?`, and builds words from runs of letters, digits, and apostrophes. On the provided input the plain Python tokenizer finds 9 sentences and 95 words. The library tokenizers are expected to agree on the sentence count for this simple text while reporting a higher token count, since NLTK and spaCy keep punctuation marks as separate tokens whereas the plain version discards them.

### 2.3 Stop word removal

The third text is tokenized with NLTK, punctuation tokens are dropped with `isalpha()`, and each remaining token is checked against NLTK's English stop word list case-insensitively. The notebook reports the original tokens, the filtered tokens, the removed stop words, and a frequency count of the removed words. The input text is itself about stop words and is dense with them (the, is, a, an, of, and, to), so the filtered version is substantially shorter, illustrating why stop word removal reduces text size for downstream NLP tasks.

### Conclusion

The experiment walks through the canonical preprocessing chain, normalize, tokenize, filter. Comparing NLTK, spaCy, and a hand-written tokenizer shows that libraries handle punctuation and sentence boundaries more precisely, while the plain Python versions make the underlying logic explicit.

## Overall Summary

Experiment 1 covers structured data analysis with pandas, grouping, filtering, and plotting a tabular dataset. Experiment 2 covers unstructured text preparation for NLP, cleaning, tokenization, and stop word removal. Together they exercise both halves of a typical data science workflow: numeric exploration of tabular data and preprocessing of raw text.
