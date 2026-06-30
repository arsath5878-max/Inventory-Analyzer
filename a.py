import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title="AI Demand Forecasting", layout="wide")
st.title("📦 AI Store Item Demand Forecasting System")
uploaded_file = st.file_uploader("Upload Store Item Demand Dataset",type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # here new features are extracted
    data["date"] = pd.to_datetime(data["date"])

    data["Year"] = data["date"].dt.year
    data["Month"] = data["date"].dt.month
    data["Day"] = data["date"].dt.day
    data["DayOfWeek"] = data["date"].dt.dayofweek
    data["WeekOfYear"] = data["date"].dt.isocalendar().week.astype(int)

  
    item_encoder = LabelEncoder()
    data["item"] = item_encoder.fit_transform(data["item"])
    data = data.sort_values(["store","item","date"])
   
    
    data["Yesterday_Sales"] = data.groupby(
        ["store","item"]
    )["sales"].shift(1)

    data["Last_Week_Sales"] = data.groupby(
        ["store","item"]
    )["sales"].shift(7)

    data["Last_Month_Sales"] = data.groupby(
        ["store","item"]
    )["sales"].shift(30)
    # -----------------------------------
    # Rolling Average
    # average of last 7 days
    # average of last 30 days

    data["Rolling_7"] = data.groupby(["store","item"])["sales"].transform(lambda x: x.shift(1).rolling(7).mean())

    data["Rolling_30"] = data.groupby(["store","item"])["sales"].transform(lambda x: x.shift(1).rolling(30).mean())

    data = data.dropna()

    # -----------------------------------
    # Features
    # -----------------------------------
    X = data[
     [
         "store",
         "item",
         "Year",
         "Month",
         "Day",
         "DayOfWeek",
         "WeekOfYear",
         "Yesterday_Sales",
         "Last_Week_Sales" ,
         "Last_Month_Sales",
         "Rolling_7",
         "Rolling_30"
     ]
 ]
    y = data["sales"]

    
    # Train/Test Split
    

    split = int(len(data)*0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    
    # Train Model
    

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    # Predict entire dataset
    data["Next_Day_Demand"] = model.predict(X)

# Calculate Reorder Qty
    data["Reorder_Qty"] = (
    data["Next_Day_Demand"] - data["sales"]
   ).clip(lower=0).astype(int)


# Health Score
    data["Health_Score"] = (
    data["sales"] /(data["Next_Day_Demand"] + 1)) * 100
    mae = mean_absolute_error(y_test,pred)

    st.sidebar.success("Model Trained")

    st.sidebar.metric(
        "MAE",
        round(mae,2)
    )
    if st.button("predict_given_dataset"):
       st.subheader("📊 Predicted Demand for Entire Dataset")
       data["Product"] = item_encoder.inverse_transform(data["item"])
       st.dataframe(
        data[
            [
                "date",
                "store",
                "Product",
                "sales",
                "Next_Day_Demand",
                "Reorder_Qty",
                "Health_Score"
            ]
        ]
    )

    st.subheader("Predict Future Demand")

    store = st.number_input(
        "Store",
        min_value=1,
        max_value=10,
        value=1
    )

    product_name = st.selectbox(
    "Select Product",
    sorted(item_encoder.classes_)
)

    item = item_encoder.transform([product_name])[0]

    year = st.number_input(
        "Year",
        value=2022
    )

    month = st.number_input(
         "Month",
        min_value=1,
        max_value=12,
        value=1
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1
    )

    weekday = st.number_input(
        "Day Of Week (0=Mon)",
        min_value=0,
        max_value=6,
        value=0
    )

    week = st.number_input(
        "Week Of Year",
        min_value=1,
        max_value=53,
        value=1
    )
    Yesterday_Sales = st.number_input(
        "Yesterday Sales",
        value=20
    )
    Last_Week_Sales = st.number_input(
        "Last Week Sales",
        value=18
    )

    Last_Month_Sales = st.number_input(
        "Last Month Sales",
        value=22
    )

    rolling7 = st.number_input(
        "7 Day Average",
        value=19.5
    )

    rolling30 = st.number_input(
        "30 Day Average",
        value=20.4
    )
    current_stock = st.number_input(
        "Current Stock",
        value=50
    )
   

    if st.button("Predict"):

        future = pd.DataFrame({

            "store":[store],
            "item":[item],
            "Year":[year],
            "Month":[month],
            "Day":[day],
            "DayOfWeek":[weekday],
            "WeekOfYear":[week],
            "Yesterday_Sales":[Yesterday_Sales],
            "Last_Week_Sales":[Last_Week_Sales],
            "Last_Month_Sales":[Last_Month_Sales],
            "Rolling_7":[rolling7],
            "Rolling_30":[rolling30],
            

        })

        future_sales = model.predict(future)[0]

        reorder = max(
            0,
            int(future_sales-current_stock)
        )

        st.success(
            f"Predicted Demand : {int(future_sales)} Units"
        )

        st.info(
            f"Recommended Reorder Quantity : {reorder}"
        )

        if reorder>0:
            st.warning("⚠ Reorder Required")
        else:
            st.success("✅ Stock is Sufficient")

        st.subheader("Feature Importance")


   
        trend = data.groupby("date")["sales"].sum()
        st.subheader("Demand Trend")

        st.line_chart(trend)
        col1, col2, col3 = st.columns(3)

        store_sales = data.groupby("store")["sales"].sum()
        st.subheader("Store-wise Sales")
        st.bar_chart(store_sales)

