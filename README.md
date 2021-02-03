# Workday data processing and analysis

This repo contains single-file processing modules and batch processing drivers for Ramses in the Wild and Innospark experiments

## Dependencies

1. download activity survey data from google forms
1. download raw EEG files (from google drive) and unzip

## Processing Pipeline

### EEG data files

batch process headers

```sh
python parse_front_matter.py
```

run QA and featurization _for each EEG file_:

```sh
python featurize_workday.py
```



## MANIFEST

| directory | role | description |
| ---- | ---- | ---- |
| analysis | workspace | R modules and experiment script; python prototypes |
| util | package | python util package to be sourced into top-level py scripts |

<!-- | preproc | pipeline | batch processing scripts to clean-up stroop 2020 data. see [README](preproc/README.md) |-->