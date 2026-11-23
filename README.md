# Grain elevator project: code

This repository holds all the code from a research project. The project dates when each USDA WCMD licensed grain elevator first appears in NAIP aerial photos. **Only code is included.** The data, the imagery (about 28 GB), the outputs and the trained model files are kept separately.

## How the model works
1. **Features:** a frozen, ImageNet-pretrained ResNet-50 turns each 300 m NAIP image chip into 2,048 features.
2. **Classifier:** a logistic regression on those features gives P(elevator) for each chip.
3. **Dating:** each site's yearly probabilities go into a step rule, which gives the first year the elevator is visible. That year is the first visible year, not the build year.
4. **Testing:** cross-validation is grouped by site, so no site appears in both training and test.
5. **Town search:** the same detector scans a town for elevators. It suggests 4 spots, and a person picks the right one.

`finetune.py` is the first test of fine-tuning the whole network. It is the starting point for the next baseline.

## Scripts

### Model (`src/ml/`)
| Script | What it does |
|---|---|
| `build_dataset.py` | Build the weakly labelled chip dataset |
| `train_eval.py` | Train and evaluate the presence classifier |
| `train_v2.py` | Detector v2: chips plus every labelled year image |
| `finetune.py` | Fine-tune the whole ResNet-50 on the GPU |
| `apply.py` | Apply the classifier to every site's NAIP series |
| `dating_experiment.py`, `model_search_dating.py` | Dating rule experiments, tested leaving one site out at a time |
| `image_quality.py` | Flag blank (no-data) NAIP images |
| `review_triage.py`, `review_queue.py` | Choose which ML dates a person should check |
| `date_check_page.py`, `date_check_spots.py`, `date_check_summary.py` | Human date-check page and its summary |
| `hard_negatives.py`, `tune_search.py`, `search_v4.py`, `road_buffer_search_test.py` | Train and test the town-search detector |
| `town_scan.py`, `town_candidates.py`, `town_resolve.py`, `town_pick_merge.py`, `candidate_review.py` | Town search: suggest spots, person picks |

### Pipeline (`src/`), in order
| Step | Script | What it does |
|---|---|---|
| 1 | `clean_dataset.py` | Clean the raw WCMD export and build the facility table |
| 2 | `find_addresses.py`, `company_locations.py` | Locate each facility, using company location lists and other public evidence |
| 3 | `geocode.py` | Turn street addresses into coordinates, with a quality flag |
| 4 | `earth_engine.py`, `export_yearly_imagery.py` | Download one NAIP image per facility per year (Google Earth Engine) |
| 5 | `src/ml/` | Score each year and estimate the first visible year (see above) |
| 6 | `review_facilities.py`, `review_location_check.py`, `review_index.py`, `preview_location.py` | Human review pages for locations and dates |
| 7 | `estimate_build_year.py`, `validate.py` | Turn labels into first-appearance estimates and compare them with known dates |
| 8 | `build_master.py` | Final dataset: USDA fields, location and the imagery estimates |

| Other | What it does |
|---|---|
| `run_overnight.py` | Run the whole pipeline unattended |
| `utils.py` | Config, paths, logging, caches |
| `select_prototype.py` | Pick the first ~10 test facilities |
| `road_distance_test.py` | Test how close elevators are to roads (used by the town search) |
| `explorer/build_explorer.py` | Build the time-lapse explorer page |
| `browser/extract_addresses.js` | Browser helper for copying locations from company websites |

## Setup
```
pip install -r requirements.txt
earthengine authenticate
```
Paths and settings are in `config/config.yaml`. Scripts expect the data folders (`data/`, `outputs/`) at the project root.

## Code review fixes (2026-11-22)

- `validate.py`: "present at start" sites now check a construction year as an upper bound (was reversed).
- `geocode.py`: and/or precedence fixed; person-picked and company-coordinate sources handled explicitly;
  unknown coordinate sources fail closed (NONE) instead of defaulting to REGISTRY.
- `build_master.py`: no crash without an imagery manifest.
- `ml/apply.py`: facilities used to train model_v2 are flagged as in-sample.
- `ml/train_v2.py`, `ml/dating_experiment.py`: crop size per image from the manifest, not the config.
- `ml/build_dataset.py`: hard-negative comment corrected.
