import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 

model = pickle.load(open('Model_Prediksi.sav','rb'))

df = pd.read_csv("DailyDelhiClimate.csv")
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.set_index(['date'], inplace=True)

st.title('Forecasting Temperature')
img = Image.open('Suhu.jpg')
img = img.resize((500,260))
st.image(img)
date = st.slider("Prediksi berapa hari kedepan",1,365, step=1)

pred = model.forecast(date)
pred = pd.DataFrame(pred, columns=['meantemp'])

if st.button("Prediksi"):
    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig,ax = plt.subplots()
        df['meantemp'].plot(style='--', color='gray', legend=True, label='known')
        pred['meantemp'].plot(color='b', legend=True, label='prediction')
        st.pyplot(fig)