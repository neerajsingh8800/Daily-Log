# Module 06: Sports Analytics and Exploratory Data Analysis

While business analytics focuses on revenue and customer retention, Sports Analytics shifts the focus toward performance optimization and game outcome prediction. This module acts as the crucial bridge between traditional Data Analytics and Machine Learning. 

Before feeding historical Indian Premier League (IPL) match statistics into a predictive algorithm like XGBoost, the raw data must undergo rigorous Exploratory Data Analysis (EDA) and Feature Engineering.

---

## 1. The Goal of Sports EDA

In sports data, raw box scores (runs, wickets, overs) are highly dimensional and noisy. The primary goal of EDA here is not just to build a retrospective dashboard, but to discover mathematical patterns that dictate winning or losing.

**Key EDA Objectives:**
* **Handling Missing Data:** Dealing with abandoned matches, missing ball-by-ball data, or DLS (Duckworth-Lewis-Stern) method adjustments in historical records.
* **Outlier Detection:** Identifying statistical anomalies (e.g., a bowler taking 5 wickets for 5 runs) and determining whether they represent genuine skill or random variance.
* **Target Leakage Prevention:** Ensuring that the data used to predict a match winner would actually be available *before* the match starts. You cannot use the "total runs scored in the second innings" as a feature to predict the winner before the game begins.

---

## 2. Feature Engineering & Mathematical Formulas

To prepare data for a machine learning model, you must mathematically transform raw statistics into predictive features. 

### A. Rolling Venue Averages
* **Theory:** Pitches behave differently. A score of 160 might be a guaranteed winning total at a slow, spinning venue, but a losing total at a high-scoring venue with short boundaries. We calculate the historical rolling average of the venue to give the model environmental context.
* **Formula:**

```math
V_{avg} = \frac{1}{N} \sum_{i=1}^{N} R_i
```

*(Where **N** is the number of historical matches played at the venue, and **R<sub>i</sub>** is the total first-innings runs scored in match **i**).*

### B. Player and Team Form (Recent Momentum)
* **Theory:** A team's overall historical win rate across a decade matters far less than their win rate in the last 5 matches. We engineer features to capture current momentum.
* **Formula:** (Batting Strike Rate for evaluating current form)

```math
SR = \left( \frac{\text{Runs Scored}}{\text{Balls Faced}} \right) \times 100
```

---

## 3. Implementation Example: Pre-Processing for Machine Learning

When preparing sports data for an ML pipeline (like a Streamlit web app powered by an XGBoost predictor), we typically transition from SQL to Python (Pandas). This allows us to handle the complex rolling calculations and matrix transformations efficiently.

Here is a production-grade Python script demonstrating how to clean IPL match data and engineer the required features before training.

```python
import pandas as pd
import numpy as np

# 1. Load the raw historical match data
df = pd.read_csv('ipl_matches_history.csv')

# 2. Data Cleaning: Drop matches with no result (e.g., rain abandonments)
df = df.dropna(subset=['winner'])

# 3. Feature Engineering: Calculate cumulative win percentages before the current match
def calculate_historical_win_rate(team, date, data):
    """Calculates a team's win rate strictly prior to the current match date."""
    past_matches = data[(data['date'] < date) & ((data['team1'] == team) | (data['team2'] == team))]
    
    if len(past_matches) == 0:
        return 0.5 # Default 50% probability if there is no historical data
        
    wins = len(past_matches[past_matches['winner'] == team])
    return wins / len(past_matches)

# Apply the function to engineer new columns
df['team1_win_rate'] = df.apply(lambda row: calculate_historical_win_rate(row['team1'], row['date'], df), axis=1)
df['team2_win_rate'] = df.apply(lambda row: calculate_historical_win_rate(row['team2'], row['date'], df), axis=1)

# 4. Feature Engineering: Venue Target Score
# Calculate average first innings score per venue to feed into the predictor
venue_stats = df.groupby('venue')['first_innings_runs'].mean().reset_index()
venue_stats.rename(columns={'first_innings_runs': 'venue_avg_runs'}, inplace=True)
df = df.merge(venue_stats, on='venue', how='left')

# 5. Final ML Feature Selection
# Isolate only the features known strictly BEFORE the first ball is bowled
features = ['venue_avg_runs', 'team1_win_rate', 'team2_win_rate', 'toss_winner', 'toss_decision']

X = df[features] # The predictive features
y = df['winner'] # The target variable

# The dataset (X, y) is now perfectly formatted to train an XGBoost model.
```
