# Time Series Analysis of Wind Energy Production in Türkiye

This repository contains the complete codebase for the thesis project titled ["Time Series Analysis of Wind Energy Production in Türkiye"](thesis.kutlaykizil.com). The project focuses on the creation of a comprehensive Wind Farm Database (WFD) for Türkiye and the subsequent use of this database for in-depth time series analysis and wind power forecasting using machine learning models.

## Abstract

The increasing integration of variable renewable energy sources, particularly wind power, into national grids presents significant challenges for maintaining grid stability and ensuring efficient energy market operations. Accurate forecasting of wind power generation is crucial for addressing these challenges. This thesis introduces the development of a comprehensive Wind Farm Database (WFD) for Türkiye, which consolidates data from disparate official sources, including the Energy Market Regulatory Authority (EPDK), Energy Exchange Istanbul (EPIAS), and the Ministry of Energy and Natural Resources. The WFD integrates technical specifications of wind farms, their hourly energy production, and relevant meteorological data. Utilizing this database, the study conducts a thorough analysis of wind power generation trends and capacity factors across Türkiye. Furthermore, it implements and evaluates Long Short-Term Memory (LSTM) neural network models to forecast wind power generation. The models leverage historical production data and meteorological inputs from the ERA5 reanalysis dataset to predict future energy output. The findings demonstrate that the developed LSTM models provide reliable and accurate forecasts, offering a valuable tool for energy planners, grid operators, and researchers in Türkiye. This research not only contributes a significant dataset to the public domain but also establishes a robust framework for wind energy analysis and forecasting in the region.

---
Most of the text below is written by an LLM, sorry in advance.

## Key Features

* **Comprehensive Wind Farm Database (WFD):** A unified SQLite database consolidating information from the Energy Market Regulatory Authority (EPDK), Energy Exchange Istanbul (EPIAS), the Ministry of Energy and Natural Resources, and the Turkish Wind Energy Association (TÜREB).
* **Automated Data Pipelines:** Scripts to automate the collection, cleaning, and updating of the database from various sources.
* **In-depth Time Series Analysis:** Tools and notebooks for analyzing capacity factors, production trends, and other key metrics of wind farms.
* **Advanced Forecasting Models:** Implementation of LSTM-based models, including versions with Attention mechanisms, for accurate wind power forecasting.
* **Rich Feature Integration:** The models are enriched with meteorological data from ERA5 (e.g., wind speed, air density) and topographical data from Digital Elevation Models (DEM).

## Repository Structure

The project's source code is organized into several directories, each with a specific purpose.
.
└── src/
├── DatabaseCreation/      # Scripts to build the initial database from raw data
│   ├── epdkImport/
│   ├── epiasImport/
│   └── ministryImport/
├── DatabaseUpdate/        # Scripts for periodic updates to the database
│   ├── epdkUpdate/
│   ├── epiasProdUpdate/
│   └── CapacityFactor/
├── DatabaseGetInfo/       # Notebooks and scripts for querying and plotting data
├── ThesisProject/         # Core research, analysis, and modeling code
│   ├── CF_table/          # Capacity Factor analysis and reporting
│   ├── DEM/               # Digital Elevation Model data processing
│   ├── ERA5/              # Scripts to download and process ERA5 weather data
│   └── LSTM/              # LSTM model implementation, training, and evaluation
└── ...


* **`src/DatabaseCreation`**: Contains all the necessary scripts to build the Wind Farm Database from scratch. It fetches and processes data from various sources (`.csv`, `.xlsx`) and inserts it into a structured SQLite database.
* **`src/DatabaseUpdate`**: Holds scripts designed to periodically update the database with the latest available information, ensuring the WFD remains current. This includes new production data, license information, and capacity factor calculations.
* **`src/DatabaseGetInfo`**: A collection of Python scripts and Jupyter Notebooks for interacting with the WFD. It includes a `DatabaseAnalyzer` for performing calculations and a `DatabasePlotter` for creating visualizations.
* **`src/ThesisProject`**: This is the heart of the research. It's subdivided into the main components of the thesis:
    * **`CF_table`**: Scripts for detailed analysis and reporting on the capacity factors of wind farms.
    * **`DEM` & `ERA5`**: Code for acquiring and processing external datasets—topographical and meteorological—used as features in the machine learning models.
    * **`LSTM`**: Contains the Python code for the LSTM models (`model3.py`, `model4_attention.py`), feature engineering notebooks, and scripts for analyzing model performance and hyperparameters.

## Acknowledgments

The Database file as well as the already trained models are availably upon request. 
 
I extend my deepest thanks to my supervisor, Assoc. Prof. Dr. Ferhat BİNGÖL, whose expert guidance was paramount to this thesis. His dedication to my academic growth not only provided clarity in complex areas but also instilled a deeper passion for the research.

I would also like to express my gratitude to the Turkish Wind Energy Association (TÜREB) for providing valuable data concerning wind turbine specific data of the wind power plants in Türkiye, which was instrumental in developing certain aspects of the Wind Farm Database.

Finally, I am deeply grateful for the financial support received from the Scientific and Technological Research Council of Türkiye (TÜBİTAK).
