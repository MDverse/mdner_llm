## Utilities

### 1. Correct JSON annotations

To vizualize and correct json annotations, open the notebook in `notebooks/review/correct_annotations.ipynb`.

### 2. Count entities per class for each annotation

To perform statistics on the distribution of annotations per files and class, run:

```sh
uv run count-entities --annotations-dir annotations/v3
```

> This command processes all JSON files listed, counts the number of entities per class for each annotation, and outputs a TSV file with the filename, text length, and entity counts per class. It will also produce plots with class distribution for all entities and entity distribution by class.

### 3. Quality Control Inventory of Named Entities

To generate a QC inventory of annotated entities from json files, run:

```sh
uv run build-entity-vocab \
    --texts-path results/groundtruth_paths.txt \
    --out-path results/qc_annotations/entities.tsv
```

> This command will scan all JSON annotations listed in `results/groundtruth_paths.txt`, extract all annotated entities, and compile them into a TSV file at `results/qc_annotations/entities.tsv`. The TSV file will contain columns for the entity text, label, source file, and frequency of occurrence across all annotations.

> 💡 Running a QC inventory on annotation files ensures that all entities are
> consistently aggregated and normalized. This is a crucial step for
> defining **annotation rules in molecular dynamics**, helping standardize
> formats, units, and naming conventions. The generated files can be
> explored in [`notebooks/review/qc_entity_inventory_explorer.ipynb`](notebooks/review/qc_entity_inventory_explorer.ipynb)
> and the rules are documented in [`docs/annotation_rules.md`](docs/annotation_rules.md).

### 4. Select informative annotation JSON files

To select informative annotation JSON files and export their paths in a text file, run:

```sh
uv run select-annotations \
        --annotations-dir annotations/v3 \
        --nb-files 50
```

> This command selects up to 50 annotation JSON files from `annotations/v3` according to entity coverage and recency, and writes their paths to a `.txt` file in the `results` folder. The selection is based on a scoring system that prioritizes files with more entities and more recent modification dates.

### 5. Extract description from parquet files

To extract dataset descriptions from parquet files and save them as `.txt` files, run:

```sh
uv run extract-description \
    --input-dir data/parquets \
    --output-dir annotations/v3
```

> This command reads all parquet files in `data/parquets` and creates one text file per row. For each row, it extracts the title and description from each file, and saves them as `<repository_name>_<dataset_id>.txt`  in the `annotations/v3` directory.

### 6. Transfer annotations to new JSON files

To transfer annotations from old JSON files to new ones, run:

```sh
uv run transfer-annotations \
    --old-annotations-dir annotations/v2 \
    --new-annotations-dir annotations/v3
```

> This command transfers annotations from JSON files in `annotations/v2` to corresponding files in `annotations/v3` based on filename matching. It reads each old annotation file, extracts the entities, and updates the new annotation file with these entities while preserving the original text and metadata. The updated annotations are saved back to the new annotation directory.


### 7. Validate annotations

To validate the annotations in a JSON file or a directory of JSON files, run:

#### Validate a single JSON file
```sh
uv run validate-annotations --json-path annotations/v3/figshare_121241.json
```

#### Validate all JSON files in a directory
```sh
uv run validate-annotations --annotations-dir annotations/v3
```

> This command checks the validity of annotations in the specified JSON file or all JSON files in the given directory. It performs several checks, including verifying that the annotated text matches the corresponding text span in the original text, ensuring that entity spans are valid and do not overlap, and removing unwanted entities based on a predefined list.


### 8. Plot performance metrics

To plot the evaluation metrics of a single gliner model, run:

```sh
uv run plot-metrics \
    --input-file <path-to-xlsx-file>
    --model-name <model-name>
```
> This command reads the evaluation metrics from the specified Excel file, generates and saves a plot of precision, recall, and F1-score for each entity class.