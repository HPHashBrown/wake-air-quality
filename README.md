PROJECT AIR (Atmospheric Intelligence & Response)
This dashboard serves as a real-time, data-driven web application designed to monitor and forecast PM2.5 pollution levels in Wake County, North Carolina. By retrieving the latest air quality readings and applying predictive modeling, the application provides both current atmospheric status and future trends.

Overview
This project synthesizes environmental data with machine learning to offer insight into regional air quality. It is built using Python and Streamlit, leveraging the EPA AirData API to provide live updates, and is deployed via Streamlit Cloud for public access.

Key Features
Real-Time Data Retrieval: Automatically fetches the latest PM2.5 readings from the EPA AirData API.

Automated Updates: The dataset refreshes dynamically whenever the application runs.

Historical Visualization: Displays comprehensive trends in local air quality over time.

Machine Learning Forecasting: Uses linear regression to generate a 10-year projection of PM2.5 levels.

Interactive Interface: A clean, accessible dashboard layout that requires no login.

Project Methodology
The application follows a structured data pipeline to generate insights:

Data Ingestion: The app loads historical PM2.5 records from a local CSV file (wake_pm25_by_year.csv).

Real-Time Synthesis: It fetches the current year’s daily values from the EPA API and merges them with the historical dataset.

Modeling: A linear regression model is trained on the consolidated data.

Forecasting: The model projects PM2.5 trends for the next 10 years.

Reporting: Results are rendered through interactive tables, model statistics, and comparative graphs.

Tech Stack
Language: Python 3.9+

Framework: Streamlit

Data Science: Pandas, NumPy, Scikit-Learn

Visualization: Matplotlib

API Handling: Requests

External Data: EPA AirData API

Deployment
This application is deployed using Streamlit Cloud. The production site is configured to rebuild automatically whenever updates are pushed to the GitHub repository, ensuring the data and features remain current.

Data Sources
All air quality data is retrieved from the EPA AirData API.

Parameter Code: 88101 (PM2.5)

State: 37 (North Carolina)

County: 183 (Wake County)

Author
Harshuaz

Wake Early College of Health and Sciences

Wake Forest, NC

License
This project is open-source and free to use for educational and research purposes.
