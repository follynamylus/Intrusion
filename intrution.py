import pandas as pd
import streamlit as st 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

tab_1,tab_2 = st.tabs(['VIEW PREDICTION','DATAFRAME AND DOWNLOAD'])

model = pickle.load(open("Gb.pkl", 'rb'))

option = st.sidebar.selectbox("Choose the type of prediction to perform",["Single","Multiple"])
if option.lower() == "single" :
    st.sidebar.title("Data Input")
    duration = st.sidebar.number_input("Input the Duration",0,42908)
    src_bytes = st.sidebar.number_input("Input the Source Bytes",0,10000000000)
    dst_bytes = st.sidebar.number_input("Input the Destination Bytes",0,10000000000)
    ssr = st.sidebar.number_input("Input the Same Service Rate",0.00,1.00,step=1e-2,format="%.2f")
    dsr = st.sidebar.number_input("Input the Different Service Rate",0.00,1.00,step=1e-2,format="%.2f")
    count = st.sidebar.number_input("Input the Count",0,520)
    dst_host_count = st.sidebar.number_input("Input the destination and host count",0,255)
    num_comp = st.sidebar.number_input("Input the Num Compromised",0,8000)
    wrong_fragment = st.sidebar.number_input("Input the Wrong Fragment",0,3)

    df = pd.DataFrame()

    df['duration'] = [duration]
    df['src_bytes'] = [src_bytes]
    df['dst_bytes'] = [dst_bytes]
    df['same_srv_rate'] = [ssr]
    df['diff_srv_rate'] = [dsr]
    df['count'] = [count]
    df['dst_host_count'] = [dst_host_count]
    df['num_compromised']  = [num_comp]
    df['wrong_fragment'] = [wrong_fragment]

    data = df.copy()

    pred = model.predict(data)


    if pred == 0 :
        prediction = "Attack"
    else : 
        prediction = "Normal"

    df["prediction"] = prediction

    tab_1.success("Prediction")
    tab_1.title(f"From the features provided and the threshold considered, The connection is predicted to be {prediction}")

    tab_2.success("Dataframe after prediction")
    tab_2.dataframe(df)
else :
    file = st.sidebar.file_uploader("Upload file to test")
    if file == None :
        st.write("Input File")
    else :
        df = pd.read_csv(file)

        tab_2.success("Dataframe before prediction")
        tab_2.dataframe(df)
        pred = model.predict(df)
        prediction = []
        for i in pred :
             if i == 0 :
                prediction.append("Attack")
             else : 
                prediction.append("Normal")
        df['predictions'] = prediction
        
        tab_1.success("Prediction class count table")
        tab_1.write(df['predictions'].value_counts())
    
        tab_2.success("Datafram after prediction")
        tab_2.dataframe(df)

    
        fig = sns.countplot(data = df, x= "predictions")
        plt.savefig('predicted_probabilities.png')
        tab_1.success("Count Plot for prediction")
        tab_1.pyplot()
        @st.cache_data # <------------- IMPORTANT: Cache the conversion to prevent computation on every rerun
        def convert_df(df): # <--------------- Function declaration
            '''
            Convert_df function converts the resulting dataframe to a CSV file.
            It takes in a data frame as a aprameter.
            It returns a CSV file
            '''
            return df.to_csv().encode('utf-8') # <--------------- Return dataframe as a CSV file
        csv = convert_df(df) # <------------ Convert_df function calling and assigning to a variable.
        tab_2.success("Print Result as CSV file") # <--------------- A widget as heading for the download option in tab 2
        tab_2.download_button("Download",csv,"Prediction.csv",'text/csv') # <------------------ Download button widget in tab 2.

