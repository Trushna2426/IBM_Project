# House Price Data Analysis with AI

**IBM SkillsBuild – Data Analytics with AI**  
**Programme Partner:** FutureTech / BharatCares

A data analytics project that explores a residential housing dataset to identify
patterns and relationships between property characteristics and house prices —
using Python-based data analytics supported by Generative AI (IBM Bob).

---

## Project Overview

This project analyses a dataset of 545 residential properties to answer the question:
*which property characteristics are associated with higher or lower house prices?*

The analysis uses Python (pandas, NumPy, Matplotlib) to perform data cleaning, exploratory data analysis, statistical comparisons, and correlation analysis.
Generative AI (IBM Bob) was used as an analytical support tool — to help formulate questions, understand results, and improve documentation.

**This is a Data Analytics project. It does not train a machine learning model
and does not predict house prices.**

---

## Dataset Description

**File:** `Housing.csv`  
**Records:** 545 properties  
**Features:** 13 columns

| Column | Type | Description |
|--------|------|-------------|
| `price` | Numeric | Sale price of the property (Rs) |
| `area` | Numeric | Plot area in sq ft |
| `bedrooms` | Numeric | Number of bedrooms (1–6) |
| `bathrooms` | Numeric | Number of bathrooms (1–4) |
| `stories` | Numeric | Number of floors (1–4) |
| `parking` | Numeric | Number of parking spaces (0–3) |
| `mainroad` | yes / no | Has main-road access |
| `guestroom` | yes / no | Has a guest room |
| `basement` | yes / no | Has a basement |
| `hotwaterheating` | yes / no | Has hot-water heating |
| `airconditioning` | yes / no | Has air conditioning |
| `prefarea` | yes / no | Located in a preferred area |
| `furnishingstatus` | Category | furnished / semi-furnished / unfurnished |

---

## Project Objectives

1. Understand the structure and contents of the housing dataset.
2. Check data quality — identify missing values and duplicate records.
3. Analyse the distribution of house prices.
4. Study the relationship between property features and price.
5. Compare average prices across different property categories.
6. Identify meaningful correlations among numeric features.
7. Create clear Matplotlib visualisations for each analytical finding.
8. Use Generative AI (IBM Bob) as an analytical support tool throughout the workflow.
9. Translate analytical findings into plain-language business insights.

---

## Technologies and Tools Used

| Tool / Library | Purpose |
|----------------|---------|
| Python 3 | Core programming language |
| pandas | Data loading, cleaning, groupby analysis |
| NumPy | Numerical calculations |
| Matplotlib | Charts and visualisations |
| Jupyter Notebook | Interactive analysis environment |
| IBM Bob (Generative AI) | Analytical support — questions, interpretations, documentation |
| Dataset used | https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction |

---

## Project Workflow

```
1. Load Housing.csv into a pandas DataFrame
        ↓
2. Understand the dataset — shape, columns, data types, statistics
        ↓
3. Data quality check — missing values, duplicates, category counts
        ↓
4. Exploratory Data Analysis — price distribution, summary statistics
        ↓
5. Answer 9 analytical questions using groupby, mean, median, count
        ↓
6. Binary feature comparison — grouped bar chart for all yes/no features
        ↓
7. Correlation analysis — Pearson matrix and heatmap
        ↓
8. AI-assisted interpretation — document how IBM Bob supported the workflow
        ↓
9. Key findings — dynamic calculations from the dataset
        ↓
10. Business insights — plain-language interpretation of findings
        ↓
11. Conclusion
```

---

## Analytical Questions

The notebook answers nine analytical questions using actual pandas calculations.
Each question includes a summary table, a Matplotlib chart, and an interpretation.

| # | Question |
|---|----------|
| Q1 | Does larger house area generally correspond to higher house prices? |
| Q2 | How are house prices distributed across different numbers of bedrooms? |
| Q3 | How are house prices related to the number of bathrooms? |
| Q4 | How do house prices vary according to the number of stories? |
| Q5 | How does furnishing status relate to house prices? |
| Q6 | How does the number of parking spaces relate to house prices? |
| Q7 | Do houses with air conditioning show different price patterns? |
| Q8 | Do houses in the preferred area show different price patterns? |
| Q9 | Does main-road access show a difference in house prices? |

---

## Data Analysis and Visualisations

All charts are built with Matplotlib and all values are calculated from the
dataset at runtime — no values are hard-coded.

| Visualisation | Type | Section |
|---------------|------|---------|
| House price distribution | Histogram with mean/median lines | EDA |
| Area vs house price | Scatter plot | Q1 |
| Average price by bedrooms | Bar chart | Q2 |
| Average price by bathrooms | Bar chart | Q3 |
| Average price by stories | Bar chart | Q4 |
| Average price by furnishing status | Bar chart | Q5 |
| Average price by parking spaces | Bar chart | Q6 |
| Average price — air conditioning comparison | Bar chart | Q7 |
| Average price — preferred area comparison | Bar chart | Q8 |
| Average price — main road comparison | Bar chart | Q9 |
| Binary feature overview (all 6 yes/no features) | Grouped bar chart | Section 11 |
| Correlation matrix | Annotated heatmap | Section 12 |

---

## AI / Generative AI Usage

Generative AI (IBM Bob) was used **only as an analytical support tool** —
not as a prediction model and not to generate any numerical results.

| How AI was used | Detail |
|-----------------|--------|
| Understanding the dataset | AI explained what each column represents |
| Generating analytical questions | AI suggested relevant questions to explore |
| Choosing visualisation types | AI recommended appropriate chart types |
| Explaining Python / pandas code | AI clarified groupby, agg, corr, pd.cut |
| Interpreting calculated results | AI helped frame interpretations after Python ran |
| Improving code readability | AI suggested clearer variable names and comments |
| Debugging | AI identified issues in output and code structure |
| Documentation | AI helped write the problem statement and conclusions |
| Business language | AI translated analytical findings into plain English |

> **All numerical values in the notebook come from Python running against
> the actual dataset. AI did not perform any calculations.**

---

## Key Findings

*(All values below are calculated dynamically in the notebook from Housing.csv.)*

- The dataset contains **545 properties** with **no missing values** and **no duplicate rows**.
- House prices range from approximately **Rs 17.5 Lakhs** to **Rs 1.33 Crores**.
- The price distribution is **right-skewed** — most properties are priced in the lower-to-mid range.
- **Area** shows a moderate positive association with price (Pearson r ≈ 0.54).
- **Bathrooms** show the strongest positive correlation with price among numeric features.
- **Air conditioning** is associated with a notably higher average price (~43% difference).
- **Preferred area** is associated with a significantly higher average price (~33% difference).
- **Furnishing status** shows a consistent price ordering: furnished > semi-furnished > unfurnished.
- **Main-road access** is associated with higher average prices.
- Every binary yes/no feature in the dataset shows a higher average price for the "yes" category.

---

## How to Run the Project

### Prerequisites

Python 3.8 or later is required. Install the required libraries with:

```bash
pip install -r requirements.txt
```

### Option 1 — Run the Jupyter Notebook (main deliverable)

Open and run the analysis notebook cell by cell:

```bash
jupyter notebook Trushna_HousePriceAnalysis_DataAnalyticsAI.ipynb
```

Make sure `Housing.csv` is in the **same folder** as the notebook before running.

### Option 2 — Run the companion Streamlit application

A companion interactive application is also included:

```bash
streamlit run house_price_prediction.py
```

This opens an interactive data analysis dashboard at **http://localhost:8501**.

---

## Project Structure

```
House Price Analysis/
│
├── Housing.csv                                       ← dataset (545 properties, 13 features)
├── Trushna_HousePriceAnalysis.ipynb                  ← main Data Analytics + AI notebook
├── house_price_prediction.py                         ← companion Streamlit application
├── Trushna_ProjectReport.docx                        ← project report document
├── requirements.txt                                  ← Python dependencies
└── README.md                                         ← this file
```

---

## Conclusion

This project demonstrates a complete **Data Analytics with AI** workflow applied
to a residential housing dataset.

Key property characteristics — including area, bathrooms, stories, air conditioning,
preferred location, and furnishing status — were analysed and found to show
meaningful associations with house prices.

Matplotlib visualisations made analytical patterns visible and easy to communicate.
Generative AI (IBM Bob) supported the workflow as an analytical assistant, while
all calculations and outputs were produced by Python.

This project was completed as part of the **IBM SkillsBuild – Data Analytics with AI**
programme conducted through **FutureTech / BharatCares**.
