# CLNLP

Lab experiments for data analysis and Natural Language Processing. A detailed write-up of both experiments is in [REPORT.md](REPORT.md).

## Experiment 1: Employee Dataset Analysis

`Experiment-1/Experiment-1.ipynb`

Exploratory analysis of `employee_information_100.csv` (100 employee records) using pandas and matplotlib. The notebook auto-detects the department, salary, gender, experience, and age columns, then runs ten sub-experiments:

1. Average salary per department (bar chart)
2. Employee count per department
3. Male and female percentage (pie chart)
4. Salary distribution (histogram)
5. Experience vs salary (scatter plot)
6. Top 10 highest-paid employees
7. Highest salary per department
8. Employees earning above the overall average
9. Average experience per department
10. Age distribution (histogram)

**Run it:** open the notebook in Google Colab, upload `employee_information_100.csv` to `/content/`, and run all cells.

## Experiment 2: Basic Text Preprocessing

`Experiment-2/Experiment-2.ipynb`

The first steps of an NLP pipeline, run on the three text files in the same folder:

- **2.1 Text cleaning**, lowercase the text, strip punctuation and numbers, collapse extra whitespace (`2.1_text_data.txt`)
- **2.2 Tokenization**, sentence and word tokenization three ways: NLTK, spaCy, and plain Python (`2.2_tokenization_data.txt`)
- **2.3 Stop word removal**, filter NLTK English stop words out of the tokenized text (`2.3_clean_data.txt`)

**Run it:** the first cell installs the dependencies (`nltk`, `spacy`, `en_core_web_sm`) and the second downloads the required NLTK data. Run the notebook from the `Experiment-2/` folder so the text files resolve.
