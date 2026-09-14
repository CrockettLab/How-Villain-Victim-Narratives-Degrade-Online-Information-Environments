# How Villain-Victim Narratives Degrade Online Information Environments
Materials for "How villain-victim narratives degrade online information environments"

## A Note on the Reddit and News on the Web Data files

The  Reddit and News on the Web CSV files which are called by the analysis scripts are too large for GitHub and have been uploaded as Parquet instead. To run the scripts using  the Parquet files, install the `arrow` package and replace the data-loading chunk:

```r
library(arrow)

# Reddit Analyses.qmd
data = as.data.table(read_parquet('reddit_data.parquet'))
data[, created_dt := as.character(created_dt)]

# News on Web Analyses.qmd
data = as.data.table(read_parquet('now_data.parquet'))
```

Alternatively, regenerate the CSVs and leave the scripts untouched:

```bash
duckdb -c "COPY 'reddit_data.parquet' TO 'reddit_data.csv' (FORMAT CSV, HEADER);"
duckdb -c "COPY 'now_data.parquet' TO 'now_data.csv' (FORMAT CSV, HEADER);"
```
