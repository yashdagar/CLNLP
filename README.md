# CLNLP

Lab experiments covering the two halves of a typical data science workflow: numeric exploration of tabular data with pandas, and preprocessing of raw text for Natural Language Processing.

| Experiment | Topic | Notebook |
|---|---|---|
| 1 | Employee dataset analysis | [`Experiment-1/Experiment-1.ipynb`](Experiment-1/Experiment-1.ipynb) |
| 2 | Basic text preprocessing | [`Experiment-2/Experiment-2.ipynb`](Experiment-2/Experiment-2.ipynb) |

---

## Experiment 1: Employee Dataset Analysis

**Dataset:** `employee_information_100.csv`, 100 employee records with department, salary, gender, years of experience, and age.

### Objective

Explore the employee dataset and answer ten analytical questions (1.1 to 1.10) using pandas for aggregation and matplotlib for visualization.

### Method

The notebook loads the CSV, strips whitespace from column names, and auto-detects the required columns by normalizing names and matching them against common variants (`Salary`, `AnnualSalary`, `Income`, `Pay`, and so on). If a required column cannot be found, the notebook reports which ones are missing instead of failing mid-analysis.

### Sub-experiments

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

Each sub-experiment prints a labeled console section, and four of them produce charts: the department salary bar chart, the gender pie chart, and the salary and age histograms.

### Run it

Open the notebook in Google Colab, upload `employee_information_100.csv` to `/content/`, and run all cells.

---

## Experiment 2: Basic Text Preprocessing

**Inputs:** three small English text files in `Experiment-2/`: `2.1_text_data.txt`, `2.2_tokenization_data.txt`, and `2.3_clean_data.txt`.

### Objective

Apply the first steps of an NLP pipeline: clean raw text, tokenize it into sentences and words with three different approaches, and remove stop words.

### 2.1 Text cleaning

The raw text deliberately contains noise. The notebook first counts what will be removed; on the provided input it finds:

- 32 uppercase letters
- 27 punctuation marks
- 3 numbers (2026, 1000, 12345)
- 7 runs of extra whitespace, 9 redundant characters in total

Cleaning then lowercases the text, strips punctuation and digits with character filters, and collapses whitespace with `' '.join(text.split())`. The text shrinks from 787 to 735 characters and reads as one normalized line:

> natural language processing nlp is a branch of artificial intelligence it helps computers understand and process human language...

### 2.2 Tokenization, three ways

The same paragraph is tokenized with:

1. **NLTK**: `sent_tokenize` and `word_tokenize`
2. **spaCy**: `en_core_web_sm`, iterating `doc.sents` and non-space tokens
3. **Plain Python**: sentences end at `.`, `!`, or `?`; words are runs of letters, digits, and apostrophes

On the provided input the plain Python tokenizer finds 9 sentences and 95 words. The library tokenizers agree on sentence count for this simple text but report more tokens, since NLTK and spaCy keep punctuation marks as separate tokens while the plain version discards them.

### 2.3 Stop word removal

The third text is tokenized with NLTK, punctuation tokens are dropped with `isalpha()`, and each remaining token is checked case-insensitively against NLTK's English stop word list. The notebook reports the original tokens, the filtered tokens, the removed stop words, and a frequency count of the removed words. The input is dense with stop words (the, is, a, an, of, and, to), so the filtered version is substantially shorter, showing why stop word removal reduces text size for downstream NLP tasks.

### Run it

The first cell installs the dependencies (`nltk`, `spacy`, `en_core_web_sm`) and the second downloads the required NLTK data. Run the notebook from the `Experiment-2/` folder so the text files resolve.

---

## Summary

Experiment 1 demonstrates the standard pandas exploratory workflow: loading and cleaning data, grouping and aggregating, filtering with boolean masks, and visualizing distributions and relationships. Experiment 2 walks the canonical preprocessing chain, normalize, tokenize, filter, and comparing NLTK, spaCy, and hand-written tokenizers shows that libraries handle punctuation and sentence boundaries more precisely while plain Python makes the underlying logic explicit.
