# Week-8-Python-Assignment
# CORD-19 Data Explorer

A comprehensive Streamlit web application for exploring and visualizing COVID-19 research papers from the CORD-19 dataset. This tool provides interactive visualizations and analytics to help researchers and data scientists gain insights from the metadata of COVID-19 related scientific publications.


## Features

* **Interactive Data Exploration**: Filter papers by publication year and journal
* **Visual Analytics**: Multiple visualization types including bar charts, word clouds, and frequency analysis
* **Real-time Metrics**: Key statistics about the dataset and filtered results
* **Word Frequency Analysis**: Identify most common terms in paper titles
* **Journal Analysis**: Top publishers of COVID-19 research
* **Publication Timeline**: Track research trends over time

## Installation

### Prerequisites

* Python 3.7 or higher
* pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
      git clone <repository-url>
      cd <repository-directory>

2. **Create and activate a virtual environment**
   
   # Create environment
   
      python -m venv cord19_env
   
   # Activate environment (Windows)
   
      cord19_env\Scripts\activate
   
   # Activate environment (macOS/Linux)
   
      source cord19_env/bin/activate

3. **Install requirements**
      pip install -r requirements.txt

4. **Download the dataset**
   
   * Place your `metadata.csv` file in the project root directory (The one uploaded is a sample of the metadata from [kaggle.com](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) for demo purposes)
   * The dataset should contain COVID-19 research paper metadata

## Usage

1. **Ensure the virtual environment is activated**
      cord19_env\Scripts\activate  # Windows
   
   # or
   
      source cord19_env/bin/activate  # macOS/Linux

2. **Run the application**
      streamlit run cord_data.py

3. **Access the application**
   
   * Open your web browser and go to `http://localhost:8501`
   * The app will automatically reload when you make code changes

## Application Structure

The application is organized into several main sections:

### Sidebar Filters

* **Year Range Slider**: Filter papers by publication year
* **Journal Selector**: Filter by top journals (multiselect)

### Main Dashboard

* **Key Metrics**: Total papers, time period, journals, average abstract length
* **Visualization Tabs**:
  * Publications Timeline: Bar chart of papers by year
  * Top Journals: Horizontal bar chart of most prolific journals
  * Title Word Cloud: Visual representation of common words in titles
  * Top Sources: Distribution of papers by source
  * Word Frequency: Bar chart of most frequent words in titles

### Data Exploration

* Interactive data tables showing sample papers and statistics
* Journal and yearly statistics tables

## Dataset Information

The application uses the CORD-19 (COVID-19 Open Research Dataset) metadata, which includes:

* **Title**: Paper titles
* **Abstract**: Paper abstracts (word count analysis)
* **Journal**: Publication journal names
* **Publish Time**: Publication dates
* **Source**: Data source information

## Data Processing

The application performs several data cleaning and preparation steps:

1. **Missing Value Handling**: Drops columns with >50% missing values
2. **Date Conversion**: Converts publish_time to datetime format
3. **Feature Engineering**: Extracts publication year and abstract word count
4. **Text Processing**: Filters stop words and analyzes word frequencies

## Dependencies

The project requires the following Python packages (included in requirements.txt):

* `pandas`: Data manipulation and analysis
* `matplotlib`: Plotting and visualization
* `seaborn`: Statistical data visualization
* `wordcloud`: Word cloud generation
* `scikit-learn`: Text processing and stop words
* `streamlit`: Web application framework
* `numpy`: Numerical computations

## Customization

You can customize the application by:

* Modifying the color palettes in visualization functions
* Adjusting the number of top items displayed in charts
* Adding new filters or visualization types
* Changing the layout and styling of the Streamlit components

## Troubleshooting

### Common Issues

1. **File Not Found Error**: Ensure `metadata.csv` is in the correct directory
2. **Module Import Errors**: Verify all dependencies are installed correctly
3. **Memory Issues**: The dataset is large; consider filtering or sampling if needed

### Getting Help

If you encounter issues:

1. Check that your virtual environment is activated
2. Verify all packages are installed correctly
3. Ensure the CSV file is accessible and in the right format

## License

This project is intended for educational and research purposes. Please ensure compliance with the CORD-19 dataset's terms of use.

## Acknowledgments

* CORD-19 dataset providers
* Streamlit for the web application framework
* The research community for COVID-19 related work

* * *

**Note**: This application is designed for exploratory data analysis and should be used as a starting point for more detailed COVID-19 research investigations.
