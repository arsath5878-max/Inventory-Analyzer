import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from io import BytesIO

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📦",
    layout="wide"
)

st.title("📦“¦ StockSense AI")
st.subheader("Smart Inventory Demand Forecasting System")

uploaded_file = st.file_uploader(
    "Upload Sales CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    required_columns = [
        "Date",
        "Product_ID",
        "Product_Name",
        "Units_Sold",
        "Current_Stock",
        "Price"
    ]

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    if len(missing_cols) > 0:
        st.error(f"Missing Columns: {missing_cols}")

    else:

        df["Date"] = pd.to_datetime(df["Date"])

        df["Units_Sold"] = pd.to_numeric(
            df["Units_Sold"],
            errors="coerce"
        ).fillna(0)
        df["Current_Stock"] = pd.to_numeric(
            df["Current_Stock"],
            errors="coerce"
        ).fillna(0)
        df["Price"] = pd.to_numeric(
            df["Price"],
            errors="coerce"
        ).fillna(0)

        df = (
            df.sort_values("Date")
            .groupby(
                ["Date", "Product_ID", "Product_Name"],
                as_index=False
            )
            .agg(
                Units_Sold=("Units_Sold", "sum"),
                Current_Stock=("Current_Stock", "last"),
                Price=("Price", "last")
            )
        )

        df["Day"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year
        df["Weekday"] = df["Date"].dt.weekday

        df["Product_Code"] = (
            df["Product_ID"]
            .astype("category")
            .cat
            .codes
        )

        df = df.sort_values(["Product_Code", "Date"])

        min_rows_per_product = int(
            df.groupby("Product_Code")["Date"].size().min()
        )

        lag_candidates = [1, 2, 3, 7, 14]
        max_usable_lag = max(min_rows_per_product - 2, 1)
        lag_days = [l for l in lag_candidates if l <= max_usable_lag]

        for lag in lag_days:
            df[f"Sales_Lag_{lag}"] = df.groupby(
                "Product_Code"
            )["Units_Sold"].shift(lag)

        window_candidates = [3, 7, 14]
        max_window = max(min_rows_per_product - 1, 2)
        roll_windows = [w for w in window_candidates if w <= max_window]

        for window in roll_windows:
            df[f"Sales_RollMean_{window}"] = df.groupby(
                "Product_Code"
            )["Units_Sold"].transform(
                lambda s, window=window: s.shift(1).rolling(
                    window,
                    min_periods=1
                ).mean()
            )

            df[f"Sales_RollStd_{window}"] = df.groupby(
                "Product_Code"
            )["Units_Sold"].transform(
                lambda s, window=window: s.shift(1).rolling(
                    window,
                    min_periods=2
                ).std()
            )

        df["Target_Units_Sold"] = df.groupby(
            "Product_Code"
        )["Units_Sold"].shift(-1)

        feature_cols = [
            "Product_Code",
            "Current_Stock",
            "Price",
            "Day",
            "Month",
            "Year",
            "Weekday",
            *[f"Sales_Lag_{lag}" for lag in lag_days],
            *[f"Sales_RollMean_{w}" for w in roll_windows],
            *[f"Sales_RollStd_{w}" for w in roll_windows],
        ]

        model_df = df.loc[
            df["Target_Units_Sold"].notna()
        ].copy()

        unique_dates = sorted(model_df["Date"].unique())

        if len(model_df) < 3 or len(unique_dates) < 2:
            use_fallback = True
        else:
            use_fallback = len(unique_dates) < 5 or len(model_df) < 25

        split_idx = int(len(unique_dates) * 0.8)
        split_idx = max(split_idx, 1)
        train_dates = set(unique_dates[:split_idx])

        train_mask = model_df["Date"].isin(train_dates)

        X_train = model_df.loc[train_mask, feature_cols]
        y_train = model_df.loc[train_mask, "Target_Units_Sold"]

        X_test = model_df.loc[~train_mask, feature_cols]
        y_test = model_df.loc[~train_mask, "Target_Units_Sold"]

        if use_fallback:
            st.sidebar.header("Model Performance")
            st.sidebar.metric(
                "Forecast Mode",
                "History-based baseline"
            )
        else:
            model = XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist",
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            predictions = np.clip(predictions, 0, None)

            mae = mean_absolute_error(y_test, predictions)

            st.sidebar.header("Model Performance")
            st.sidebar.metric(
                "Next-day MAE",
                round(mae, 2)
            )

        st.write("## Dashboard")

        total_products = df["Product_ID"].nunique()
        total_stock = int(df["Current_Stock"].sum())
        total_sales = int(df["Units_Sold"].sum())

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Products",
            total_products
        )

        col2.metric(
            "Total Inventory",
            total_stock
        )

        col3.metric(
            "Total Sales",
            total_sales
        )

        latest_df = (
            df.sort_values(["Product_Code", "Date"])
            .groupby("Product_Code", as_index=False)
            .tail(1)
            .copy()
        )

        if len(latest_df) == 0:
            st.error(
                "Not enough per-product history to generate next-day forecasts. "
                "Add more dates per product and try again."
            )
            st.stop()

        if use_fallback:
            baseline_cols = [
                c for c in [
                    "Sales_RollMean_14",
                    "Sales_RollMean_7",
                    "Sales_RollMean_3",
                    "Sales_Lag_7",
                    "Sales_Lag_3",
                    "Sales_Lag_1",
                ]
                if c in latest_df.columns
            ]

            baseline = None

            if len(baseline_cols) > 0:
                baseline = latest_df[baseline_cols].mean(
                    axis=1,
                    skipna=True
                )

            if baseline is None:
                baseline = latest_df["Units_Sold"]
            else:
                baseline = baseline.where(
                    baseline.notna(),
                    latest_df["Units_Sold"]
                )

            latest_df["Predicted_Demand"] = baseline.fillna(0)
            latest_df["Predicted_Demand"] = np.clip(
                latest_df["Predicted_Demand"],
                0,
                None
            )
        else:
            latest_df["Predicted_Demand"] = model.predict(
                latest_df[feature_cols]
            )
            latest_df["Predicted_Demand"] = np.clip(
                latest_df["Predicted_Demand"],
                0,
                None
            )

        latest_df["Health_Score"] = (
            latest_df["Current_Stock"]
            /
            (latest_df["Predicted_Demand"] + 1)
        ) * 100

        latest_df["Recommended_Reorder"] = np.where(
            latest_df["Current_Stock"]
            <
            latest_df["Predicted_Demand"],
            (
                latest_df["Predicted_Demand"]
                -
                latest_df["Current_Stock"]
            ).astype(int),
            0
        )

        def inventory_status(score):
            if score >= 80:
                return "Healthy"
            elif score >= 50:
                return "Moderate"
            else:
                return "Risk"

        latest_df["Status"] = latest_df[
            "Health_Score"
        ].apply(inventory_status)

        st.write("## Demand Forecast (Next Day)")

        result_cols = [
            "Product_Name",
            "Current_Stock",
            "Predicted_Demand",
            "Health_Score",
            "Recommended_Reorder",
            "Status"
        ]

        st.dataframe(
            latest_df[result_cols]
        )

        st.write("## Product Analysis")

        product = st.selectbox(
            "Select Product",
            latest_df["Product_Name"].unique()
        )

        product_data = latest_df[
            latest_df["Product_Name"] == product
        ]

        current_stock = int(
            product_data["Current_Stock"].iloc[0]
        )

        predicted = int(
            product_data["Predicted_Demand"].iloc[0]
        )

        reorder = int(
            product_data["Recommended_Reorder"].iloc[0]
        )

        health = round(
            product_data["Health_Score"].iloc[0],
            2
        )

        status = product_data["Status"].iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current Stock",
            current_stock
        )

        c2.metric(
            "Predicted Demand",
            predicted
        )

        c3.metric(
            "Health Score",
            f"{health}%"
        )

        c4.metric(
            "Reorder Qty",
            reorder
        )

        st.write(f"### Status: {status}")

        if reorder > 0:
            st.warning(
                f" Reorder {reorder} units immediately."
            )
        else:
            st.success(
                " Inventory level is sufficient."
            )

        st.write("## Sales Trend")

        sales_chart = px.line(
            df,
            x="Date",
            y="Units_Sold",
            color="Product_Name",
            title="Sales Trend"
        )

        st.plotly_chart(
            sales_chart,
            use_container_width=True
        )

        st.write("## Inventory Health Distribution")

        pie = px.pie(
            latest_df,
            names="Status",
            title="Inventory Status"
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

        st.write("## Top Selling Products")

        top_products = (
            df.groupby("Product_Name")["Units_Sold"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        bar = px.bar(
            top_products,
            x="Product_Name",
            y="Units_Sold",
            title="Top Selling Products"
        )

        st.plotly_chart(
            bar,
            use_container_width=True
        )

        st.write("## AI Recommendations")

        risky_products = latest_df[
            latest_df["Recommended_Reorder"] > 0
        ]

        if len(risky_products) > 0:

            for _, row in risky_products.iterrows():

                st.error(
                    f"""
                    Product: {row['Product_Name']}
                    
                    Current Stock: {int(row['Current_Stock'])}
                    
                    Predicted Demand: {int(row['Predicted_Demand'])}
                    
                    Recommended Reorder:
                    {int(row['Recommended_Reorder'])} units
                    """
                )

        else:
            st.success(
                "All products have healthy inventory levels."
            )

        st.write("## Download Forecast Report")

        output = BytesIO()

        latest_df.to_csv(
            output,
            index=False
        )

        st.download_button(
            label="Download CSV Report",
            data=output.getvalue(),
            file_name="forecast_report.csv",
            mime="text/csv"
        )

else:

    st.info(
        "Upload a sales dataset to begin forecasting."
    )
