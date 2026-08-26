# Datasets Directory

This directory stores benchmark datasets and their corresponding SQLite databases for ModelBench. Large database files (`*.db`, `*.sqlite`) are ignored by git.

## Spider Dataset Setup

ModelBench evaluates Text-to-SQL performance using the official Spider dataset. For our experiments, we used the dataset mirror available on Kaggle.

1. **Download**: Download the Spider dataset from [Kaggle](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset) (or the [official website](https://yale-lily.github.io/spider)).
2. **Extract**: Extract the downloaded archive.
3. **Place**: Move the contents directly into the `datasets/spider/` directory.

### Expected Directory Structure

After extraction, your `datasets/` directory must look exactly like this for the pipeline to function correctly:

```text
datasets/
├── README.md
└── spider/
    ├── train_spider.json          # 7000+ training examples (used for retrieval)
    ├── train_others.json
    ├── dev.json                   # 1034 validation examples (used for evaluation)
    ├── tables.json                # Database schemas and column metadata
    └── database/                  # The SQLite databases themselves
        ├── concert_singer/
        │   ├── concert_singer.sqlite
        │   └── schema.sql
        ├── dog_kennels/
        │   ├── dog_kennels.sqlite
        │   └── schema.sql
        └── ... (160+ other database folders)
```
