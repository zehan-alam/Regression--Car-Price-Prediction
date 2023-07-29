import streamlit as st
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def outlier_remover(data,col):
    Q1 = data[col].quantile(.25)
    Q3 = data[col].quantile(.75)
    IQR = Q3-Q1
    lower_fence = Q1- 1.5*IQR
    upper_fence = Q3+ 1.5*IQR
    return data[(data[col]<upper_fence) & (data[col]>lower_fence)]

car = pd.read_csv('cardata_prediction.csv')
car = outlier_remover(car,'Selling_Price')
car = outlier_remover(car,'Present_Price')
car = outlier_remover(car,'Kms_Driven')
car['Transmission'] = car['Transmission'].replace(['Manual','Automatic'],[0,1])
car['Seller_Type'] = car['Seller_Type'].replace(['Dealer','Individual'],[0,1])
car['Age'] = datetime.date.today().year - car['Year']
car = car.drop(columns=['Car_Name','Year']).copy()
car = pd.get_dummies(car)
X = car.drop(columns='Selling_Price').values
y = car['Selling_Price'].values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
regressor = RandomForestRegressor(n_estimators= 300,min_samples_split= 2,min_samples_leaf= 1,max_depth= 5)
regressor.fit(X,y)

def predict_selling_price(present_price, kms_driven, seller_type, transmission, owner, year, fuel_type_cng, fuel_type_diesel, fuel_type_petrol):
    age = datetime.date.today().year - year
    X_new = [present_price, kms_driven, seller_type, transmission, owner, age,
             fuel_type_cng, fuel_type_diesel, fuel_type_petrol]
    X_new_scaled = scaler.transform([X_new])
    selling_price = regressor.predict(X_new_scaled)
    return selling_price[0]

def main():
    st.title('Car Selling Price Prediction')

    present_price = st.number_input('Present Price (in thousands)', min_value=0.0, value=12.5, step=0.01)
    kms_driven = st.number_input('Kilometers Driven', min_value=0, value=9000, step=1000)
    seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    fuel_type = st.selectbox('Fuel Type', ['CNG', 'Diesel','Petrol'])
    owner = st.number_input('Number of Previous Owners', min_value=0, value=0)
    year = st.number_input('Year', min_value=1950, max_value=datetime.date.today().year, value=2017)

    seller_type = 1 if seller_type == 'Individual' else 0
    transmission = 1 if transmission == 'Automatic' else 0
    fuel_type_cng = 0
    fuel_type_diesel = 0
    fuel_type_petrol = 0
    if fuel_type=='CNG':
        fuel_type_cng = 1
    elif fuel_type=='Diesel':
        fuel_type_diesel = 1
    else:
        fuel_type_petrol = 1

    if st.button('Predict Selling Price'):
        selling_price = predict_selling_price(present_price, kms_driven, seller_type, transmission, owner, year, fuel_type_cng, fuel_type_diesel, fuel_type_petrol)
        st.success(f'Predicted Selling Price: ${selling_price.round(2)}k')

if __name__ == '__main__':
    main()
