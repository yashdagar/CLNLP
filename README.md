# CLNLP

Lab experiments covering the two halves of a typical data science workflow: numeric exploration of tabular data with pandas, and preprocessing of raw text for Natural Language Processing.

| Experiment | Topic | Notebook |
|---|---|---|
| 1 | Employee dataset analysis | [`Experiment-1/Experiment-1.ipynb`](Experiment-1/Experiment-1.ipynb) |
| 2 | Basic text preprocessing | [`Experiment-2/Experiment-2.ipynb`](Experiment-2/Experiment-2.ipynb) |
| 3 | Stemming, lemmatization and regular expressions | [`Experiment-3/Experiment-3.ipynb`](Experiment-3/Experiment-3.ipynb) |
| 4 | Term frequency, NER and TF-IDF | [`Experiment-4/Experiment-4.ipynb`](Experiment-4/Experiment-4.ipynb) |
| 5 | Subword tokenization and POS tagging | [`Experiment-5/Experiment-5.ipynb`](Experiment-5/Experiment-5.ipynb) |

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

## Experiment 3: Stemming, Lemmatization and Regular Expressions

**Inputs:** three data files in `Experiment-3/`: `3.1_stemming_data.txt` (8 words), `3.2_lemmatization_data.txt` (14 words), and `3.3_regex_data.txt`, a short workshop announcement laced with emails, URLs, phone numbers, hashtags and mentions.

### Objective

Reduce words to their root forms with the two standard normalization techniques, stemming and lemmatization, compare the two on the same kind of vocabulary, and extract structured entities from free text with regular expressions.

### 3.1 Stemming

The Porter stemmer runs over the 8-word list. Stemming is pure suffix stripping, and the output shows both what that buys and what it costs: `playing`, `played` and `plays` all collapse to `play` and `connected` and `connection` correctly share `connect`, but `studies` and `studying` both become `studi` and `computers` becomes `comput`, stems that are not English words at all.

### 3.2 Lemmatization

`WordNetLemmatizer` looks roots up in a dictionary, but it assumes every word is a noun unless told otherwise. The notebook tags each word with `nltk.pos_tag` first, maps the Penn tag to WordNet's four classes, and prints a table comparing the default noun-mode lemma against the POS-aware one:

- With POS information, `running` and `studying` resolve to `run` and `study` (default mode leaves them untouched), and where the stemmer produced `studi`, the lemmatizer returns the dictionary form `study`.
- Irregular forms resolve correctly: `went` to `go`, `mice` to `mouse`, `children` to `child`, and the comparative `better` to `well`.
- The same table shows the technique's weak spot: each word is tagged in isolation, so `ran` and `ate` are mistaken for nouns and pass through unchanged. Lemmatization is only as good as the POS information it is given.

### 3.3 Regular expression extraction

Five patterns run over the announcement text and extract 2 emails, 3 URLs, 2 mobile numbers, 3 hashtags and 2 mentions. Two of the patterns need care to keep out of each other's way: the mention pattern uses a negative lookbehind so the `@gmail` inside `nlpworkshop@gmail.com` does not count as a mention, and URL matches are stripped of trailing punctuation because `\S+` cannot know that the final period of `www.python.org.` ends the sentence rather than the address.

### Run it

The first cell installs `nltk` and the second downloads WordNet and the POS tagger data. Run the notebook from the `Experiment-3/` folder so the three data files resolve.

---

## Experiment 4: Term Frequency, NER and TF-IDF

**Inputs:** `Experiment-4/input_tf_ner_data.txt`, a 301-word history of NLP written to be dense with named entities (people, organizations, dates, places, money), plus three one-sentence documents defined inline for the TF-IDF part.

### Objective

Count term frequencies with a toolkit and again with nothing but string operations, extract named entities with spaCy, and then compute TF, DF, IDF and TF-IDF entirely from scratch across three documents.

### 4.1 Term frequency and NER using toolkits

**NER.** spaCy's `en_core_web_sm` finds 38 entities in the corpus and the notebook prints each with its label and `spacy.explain` description. The clean hits show what the model is for: `Alan Turing` and `Noam Chomsky` as PERSON, `one billion dollars` as MONEY, `June 2017` as DATE, `French` as NORP. The misses are just as instructive for a small model: `OpenAI` is labeled GPE and the book title `Syntactic Structures` becomes an ORG.

**TF.** The corpus is tokenized with NLTK's `word_tokenize`, lowercased, filtered to alphanumeric tokens, and counted with `Counter`: 299 tokens, 196 unique terms, with `the` (17), `in` (14) and `a` (10) unsurprisingly on top. The full ranking is written to `4_1_tf.csv`.

### 4.2 Term frequency without a toolkit

The same corpus goes through plain string operations: lowercase, drop punctuation characters, `split()`, and count with a dict: 301 tokens, 198 unique terms, written to `4_2_tf.csv`. A comparison cell diffs the two rankings and finds exactly two disagreements, `gpt2` and `gpt3`. `word_tokenize` keeps `GPT-2` as a single token, which the `isalnum()` filter then discards, while the manual pipeline deletes the hyphen and produces `gpt2`. Same text, two defensible answers, which is why a TF pipeline has to document its tokenization.

### 4.3 TF-IDF without a toolkit

Three one-sentence documents about NLP, AI and machine learning are processed by hand: raw-count TF per document, DF across documents, IDF as log10(N/DF) with N = 3, and TF-IDF as their product. The 18-term vocabulary splits into exactly two IDF bands: terms in one document score 0.4771 and terms in two score 0.1761, and the notebook prints the full TF, DF, IDF and TF-IDF table for every document plus the top terms of each.

The top terms are the distinguishing words: `field` for document 1, `helps`, `computers`, `understand` and `human` for document 2, `machine` and `learning` for document 3, while the shared NLP vocabulary (`natural`, `language`, `processing`, `artificial`, `intelligence`) sinks to the low band. The ranking also promotes `a` and `an` simply because each happens to be unique to one document, a reminder that TF-IDF measures distinctiveness across the collection, not importance within the language.

### Run it

The first cell installs `nltk` and `spacy` with `en_core_web_sm`, and the second downloads the NLTK tokenizer data. Run the notebook from the `Experiment-4/` folder so `input_tf_ner_data.txt` resolves; the two CSV files are written next to the notebook.

---

## Experiment 5: Subword Tokenization and POS Tagging

**Input:** `Experiment-5/input_sub_word_data.txt`, a 947 word corpus about tokenization, BPE, and subword units. It spans 66 sentences and 365 unique lowercase word forms, and deliberately ends with a list of long rare words (`naturalization`, `internationalization`, `computationally`, `hydrokinetic`, and others) so that subword behaviour is visible.

### Objective

Where Experiment 2 stopped at word-level tokens, this one goes below the word: split text into subword units with BPE and SentencePiece, both with a pretrained model and with a model trained from the corpus itself, then tag parts of speech with spaCy and NLTK.

Two sentences are used throughout so the four tokenizers can be compared on the same input: the first sentence of the corpus, and the rare word sentence near the end.

### 5.1 Byte Pair Encoding

**a. Pretrained.** GPT-2's tokenizer through `AutoTokenizer`, a 50,257 piece BPE vocabulary. The first sentence becomes 22 tokens, almost all whole words carrying the `Ġ` leading-space marker. The rare word sentence needs 44, splitting `internationalization` into `Ġinternational` and `ization`.

**b. From scratch.** `learn_bpe` implements the algorithm directly. Every word starts as a character sequence with an `</w>` end-of-word marker, adjacent symbol pairs are counted across the whole corpus, the most frequent pair is merged, and the frequencies are recalculated. After 300 merges the learned vocabulary holds 278 symbols. The first merges are exactly the ones the corpus statistics predict: `e</w>`, `s</w>`, `en`, `in`, `d</w>`, `er`, then later `ing` and `ation`.

`encode_word` applies the merges to new words in the order they were learned, so unseen words still decompose into known pieces:

| Word | Learned segmentation |
|---|---|
| `naturalization` | `nat ur alization</w>` |
| `internationalization` | `internation alization</w>` |
| `computationally` | `comput ation ally</w>` |
| `hydrokinetic` | `h y d r o k in e t ic </w>` |

Merge count controls granularity directly, measured as average tokens per corpus word:

| Merges | Tokens per word |
|---|---|
| 50 | 5.30 |
| 100 | 4.50 |
| 200 | 3.67 |
| 300 | 3.11 |
| 500 | 2.31 |

Frequent words such as `subword` and `tokenization` collapse into single tokens, while `hydrokinetic`, which never appears in the training corpus in a reusable form, stays close to characters. That is the intended trade-off: a vocabulary of 278 symbols still covers words it has never seen.

### 5.2 SentencePiece

**a. Pretrained.** ALBERT's SentencePiece model, 30,000 pieces, using `▁` to mark word starts instead of GPT-2's `Ġ`. It matches GPT-2 at 22 tokens on the first sentence and needs 48 on the rare word sentence.

**b. From scratch.** `SentencePieceTrainer.train` runs on the raw input file with `vocab_size=500` and `model_type='bpe'`. Unlike the hand-written implementation, SentencePiece works on raw bytes, so it keeps casing and punctuation and needs no pre-tokenization step. The same first sentence now takes 45 tokens rather than 22, because a 500 piece vocabulary trained on 947 words cannot afford whole-word entries for much beyond the most common terms: `Natural` becomes `▁N` + `atural`.

### 5.3 POS tagging

Both taggers run on `The young student is reading an interesting book in the library.` and print unique tokens with their tag and a description.

- **spaCy** reports the coarse `pos_`, the fine-grained Penn `tag_`, and `spacy.explain(tag_)`.
- **NLTK** uses `pos_tag` over `word_tokenize`, with descriptions loaded from the bundled `upenn_tagset` help data.

They agree on every token. The one place their labels differ in kind is `is`: spaCy calls it `AUX` at the coarse level while both assign the same fine tag `VBZ`, since NLTK's Penn tagset has no separate auxiliary category.

### 5.4 POS tagging with frequency

spaCy runs over the whole corpus: 1,110 tokens, 408 unique token and tag combinations, printed with frequency and tag description. A second table aggregates by coarse tag and shows the profile of expository technical prose, nouns dominating at 298, then punctuation at 152, verbs at 130, and adjectives at 113.

### Run it

The first cell installs `transformers`, `sentencepiece`, `spacy`, and `en_core_web_sm`, and the second downloads the NLTK tokenizer, tagger, and tagset data. Run the notebook from the `Experiment-5/` folder so `input_sub_word_data.txt` resolves. The pretrained tokenizers are fetched from Hugging Face on first use, and the SentencePiece cell writes `sp_corpus.model` and `sp_corpus.vocab` next to the notebook.

---

## Summary

Experiment 1 demonstrates the standard pandas exploratory workflow: loading and cleaning data, grouping and aggregating, filtering with boolean masks, and visualizing distributions and relationships. Experiment 2 walks the canonical preprocessing chain, normalize, tokenize, filter, and comparing NLTK, spaCy, and hand-written tokenizers shows that libraries handle punctuation and sentence boundaries more precisely while plain Python makes the underlying logic explicit. Experiment 3 continues normalization below the surface form: stemming trades correctness for speed by chopping suffixes, lemmatization recovers true dictionary roots but only when it knows the part of speech, and regular expressions handle the entities that have structure rather than morphology. Experiment 4 turns tokens into numbers: term frequency computed with and without a toolkit agrees except where tokenization decisions differ, spaCy's NER shows both the reach and the limits of a small pretrained model, and the from-scratch TF-IDF makes the weighting formula concrete enough to see why shared words sink and distinctive words rise. Experiment 5 continues that chain one level down: subword tokenization removes the unknown word problem that word-level tokenization runs into, and implementing BPE by hand next to GPT-2 and SentencePiece shows that the vocabulary is not designed but learned from corpus statistics, with merge count as the single dial between character-level and word-level behaviour. POS tagging then attaches the first layer of linguistic structure on top of those tokens.
