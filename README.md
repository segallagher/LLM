# LLM
A location for my LLM practice

# Data
Data is to be stored in the `/data` directory. 
The raw data is to be stored in `/data/raw` and should be in the form of a csv file inside there
## Dataset
This project uses the Medical Meadow Medical Flashcards dataset available at [https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards/viewer/default/train?row=98](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

## Data Processing
Data can be processed by setting the `GENERATE_TOKENIZED_DATA` flag to `True`
- This will also generate a new vocabulary

If `GENERATE_TOKENIZED_DATA` is set to `False`, assuming default names, the program will attempt to load
- `test.csv`, `val.csv`, `train.csv`, `vocab.json`

# Training


# requirements
[CUDA version 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
