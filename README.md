# Computer Vision Fall 2024

### Final Project Use

To use the Final Project in its entirety, you are going to need to:

- run 'git clone https://github.com/cbn0006/COMP6970.git'
- get an API key from Polygon.io
- navigate to the Data Procurement directory
- edit the getStockData.py script's main method to include your API key and run that script
- run cleaning.py (ensure the symbol is the same in both python scripts)
- run numericalDataLabeling.py (ensure original_csv is the right csv)
- run 20MinuteVisuals.py (ensuring that the right csv is being pointed to)
- run numericalAdvancedStats.py (ensuring that the right csv is being pointed to)
- run all cells in the COMP 6970 Final Project.ipynb notebook.
- You can also just load in the saved weights from the Model Weights directory to save training time because training takes roughly 2 hours


Important things to Note:
- If you decide to run any part of the process above multiple times, delete all the files created by the overlapping process because this code is not meant to handle overlapping runs.
- getStockData.py creates "{symbol}_minute_data_raw.csv"
- cleaning.py turns "{symbol}_minute_data_raw.csv" into "{symbol}_minute_data_cleaned.csv"
  - This numerical csv data is what goes into the paper2.ipynb
- numericalDataLabeling.py turns "{symbol}_minute_data_cleaned.csv" into "{symbol}_minute_data_cleaned_labeld.csv"
- 20MinuteVisuals.py turns "{symbol}_minute_date_cleaned_labeled.csv" into labeled images
  - This image data is what goes into paper1.ipynb and the final project ipynb
- numericalAdvancedStats.py turns "{symbol}_minute_data_cleaned.csv" into "{symbol}_minute_data_cleaned_advanced.csv"
  - This numerical csv data is what goes into the final project ipynb for the MS-DDQN implementation
