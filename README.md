📦 StockSense AI - Smart Inventory Demand Forecasting System

 Overview

StockSense AI is a Machine Learning-powered inventory management and demand forecasting system that predicts future product demand using historical sales data. The application helps businesses optimize inventory levels, prevent stock shortages, and reduce overstock situations through intelligent forecasting and reorder recommendations.

 🎯 Features

- 📈 Demand Forecasting using XGBoost
- 📦 Inventory Health Score Calculation
- 🔄 Smart Reorder Recommendations
- ⚠️ Low Stock Risk Detection
- 📊 Interactive Dashboard using Streamlit
- 📉 Sales Trend Visualization
- 📋 Product-wise Inventory Analysis
- 📥 Excel Report Download

---

🛠️ Tech Stack

- **Python**
- **Streamlit**
- **XGBoost**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **Plotly**
- **OpenPyXL**

---

 📂 Dataset Requirements

The uploaded CSV file must contain the following columns:

| Column Name | Description |
|------------|-------------|
| Date | Sales Date |
| Product_ID | Unique Product ID |
| Product_Name | Product Name |
| Units_Sold | Quantity Sold |
| Current_Stock | Available Inventory |
| Price | Product Price |

Example:

```csv
Date,Product_ID,Product_Name,Units_Sold,Current_Stock,Price
2025-01-01,P001,Laptop,50,200,45000
2025-01-02,P001,Laptop,60,140,45000
```

 📊 Machine Learning Workflow


Sales Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Lag Features
      ↓
Rolling Statistics
      ↓
XGBoost Training
      ↓
Demand Forecasting
      ↓
Inventory Health Analysis
      ↓
Reorder Recommendation
      ↓
Interactive Dashboard
```

---

🧠 Forecasting Model

**Algorithm Used:** XGBoost Regressor

Features Used:

- Product Code
- Current Stock
- Product Price
- Day
- Month
- Year
- Weekday
- Lag Features
- Rolling Mean
- Rolling Standard Deviation

Evaluation Metric:

- Mean Absolute Error (MAE)

Expected Forecast Accuracy:

- Approximately **85% - 95%** depending on dataset quality and historical sales volume.

---

📷 Dashboard Preview

Features available in the dashboard:

- Dataset Preview
- Inventory Summary
- Demand Forecast Table
- Product Analysis
- Sales Trend Charts
- Inventory Status Distribution
- Risk Product Identification
- Download Forecast Reports

---

 📁 Project Structure

```text
StockSense-AI/
│
├── app.py
├── requirements.txt
├── sample_dataset.csv
├── README.md
│
└── screenshots/
    ├── dashboard.png
    ├── forecast.png


💡 Business Benefits

- Reduce Inventory Costs
- Prevent Stock-Out Situations
- Improve Supply Chain Decisions
- Optimize Warehouse Utilization
- Increase Customer Satisfaction
- Improve Demand Planning Accuracy



 🔮 Future Enhancements

- Multi-Day Forecasting
- Deep Learning Models (LSTM)
- Real-Time Inventory Tracking
- Cloud Deployment
- Automated Purchase Order Generation
- Email Notifications for Low Stock



👨‍💻 Author

**Arsath Ahamed**

Machine Learning & Data Science Enthusiast

link:https://inventory-analyzer-arsathlrlxlvcj94bs7jpte.streamlit.app/

