#final_project_webapp
#The streamlit link: https://cmse830finalproject-gdvn9ghmqcsiefbrjqwqvn.streamlit.app/

import streamlit as st #create a web app
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hiplot as hip
import plotly.express as px #visualize library
from PIL import Image #upload image
from sklearn.impute import KNNImputer

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR # Support Vector Regression, different from SVC (Support Vector Classifier)
from sklearn.tree import DecisionTreeRegressor
from copy import copy


##################################################################################################################################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
##################################################################################################################################################################
##define the font sizes
st.markdown(""" <style> .font_title {
font-size:45px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:40px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:28px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:24px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:22px ; font-family: 'times';text-align: justify;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text1 {
font-size:20px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext1 {
font-size:18px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True) #For figure 

st.markdown(""" <style> .font_subsubtext1 {
font-size:18px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True) #For table

st.markdown(""" <style> .font_subtext2 {
font-size:18px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True) #For picture

##################################################################################################################################################################
##title, picture and context
st.markdown('<p class="font_title">An Analysis and prediction to Beijing Air-Quality Data(2013-2017)</p>', unsafe_allow_html=True)

#upload a picture
image = Image.open('sandstorm_BJ.png')
st.image(image, use_column_width=True)#make the image take up the full column width

st.markdown('<p class="font_subtext2">Image credit: https://www.cnn.com/2023/04/11/asia/china-sandstorm-hits-beijing-intl-hnk/index.html</p>', unsafe_allow_html=True) #add empty space

st.markdown('<p class="font_text">Explore the Comprehensive Beijing Air Quality Dataset (2013-2017) from UCI education. This rich dataset features hourly recordings of air pollutants collected from 12 key air-quality monitoring sites, meticulously controlled at the national level. Delve into the interactive sidebar to select and analyze data from any of these critical monitoring sites. Spanning from March 1st, 2013, to February 28th, 2017, this dataset provides a detailed temporal snapshot of Beijing air quality. Note that missing data points are indicated as NA, allowing for a transparent and unaltered view of environmental conditions during this period. Dive into this dataset to uncover trends, patterns, and insights into Beijing air quality. </p>', unsafe_allow_html=True) #add empty space

st.markdown('<p class="font_text"></p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">The primary objectives of this study are to evaluate air quality as either satisfactory or unsatisfactory based on predefined criteria, and to thoroughly analyze the temporal distribution of pollutants. This includes examining seasonal, monthly, and daily variations. Additionally, we aim to identify the underlying causes of these trends by interpreting distribution patterns. Our methodology incorporates two key predictive techniques: Time Series Analysis (TSA) for forecasting daily pollutant levels, and Feature Selection for examining the interplay between pollutants and other variables, such as temperature. This dual approach allows for a comprehensive understanding of air quality dynamics.</p>', unsafe_allow_html=True)#add empty space

##################################################################################################################################################################

##choose the site and upload the data 
st.sidebar.markdown('<p class="font_text1">There are 12 monitering sites</p>', unsafe_allow_html=True)#choose the site(12 in total)
sites_names = ['Aoti','Gucheng','Huairou','Tiantan','Changping','Guanyuan','Nongzhanguan','Wanliu','Dongsi','Wanshou','Dingling','Shunyi']

site_name = st.sidebar.selectbox('Select a site', sites_names, index=0)

if site_name == 'Aoti':
    site_data = pd.read_csv('PRSA_Data_Aotizhongxin_20130301-20170228.csv')

if site_name == 'Gucheng':
    site_data = pd.read_csv('PRSA_Data_Gucheng_20130301-20170228.csv')

if site_name == 'Huairou':
    site_data = pd.read_csv('PRSA_Data_Huairou_20130301-20170228.csv')

if site_name == 'Tiantan':
    site_data = pd.read_csv('PRSA_Data_Tiantan_20130301-20170228.csv')

if site_name == 'Changping':
    site_data = pd.read_csv('PRSA_Data_Changping_20130301-20170228.csv')

if site_name == 'Guanyuan':
    site_data = pd.read_csv('PRSA_Data_Guanyuan_20130301-20170228.csv')

if site_name == 'Nongzhanguan':
    site_data = pd.read_csv('PRSA_Data_Nongzhanguan_20130301-20170228.csv')

if site_name == 'Wanliu':
    site_data = pd.read_csv('PRSA_Data_Wanliu_20130301-20170228.csv')

if site_name == 'Dongsi':
    site_data = pd.read_csv('PRSA_Data_Dongsi_20130301-20170228.csv')

if site_name == 'Wanshou':
    site_data = pd.read_csv('PRSA_Data_Wanshouxigong_20130301-20170228.csv')

if site_name == 'Dingling':
    site_data = pd.read_csv('PRSA_Data_Dingling_20130301-20170228.csv')

if site_name == 'Shunyi':
    site_data = pd.read_csv('PRSA_Data_Shunyi_20130301-20170228.csv')


sns.set_context("talk", font_scale=0.8)#define the size of font for all the plots 

##################################################################################################################################################################
#separate three parts
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["Information of the dataset", "Relationship Investigation", "AQG Investigation", "AQI", "Analysis", "TSA Prediction", "Feature Selection Prediction"])


##################################################################################################################################################################
##Show some basic information about the dataset
with tab1:
    st.markdown('<p class="font_text">Understanding air quality is crucial, and six pollutants stand out for their significant impact: PM2.5, PM10, Ozone (O3), Nitrogen Dioxide (NO2), Sulfur Dioxide (SO2), and Carbon Monoxide (CO) </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">PM2.5 (Particulate Matter 2.5): These fine particles, with diameters at or below 2.5 micrometers, are particularly concerning due to their ability to penetrate deep into the respiratory system. They can cause various adverse health effects, especially in the lungs and heart. </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">PM10 (Particulate Matter 10): Slightly larger than PM2.5, these particles have diameters at or below 10 micrometers. While they can enter the respiratory system, they are generally considered less hazardous than PM2.5 but still pose significant health risks.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">O3 (Ozone): At ground level, ozone is an air pollutant with harmful effects on the respiratory system. It is not emitted directly into the air but is created by chemical reactions between oxides of nitrogen (NOx) and volatile organic compounds (VOC) in the presence of sunlight.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">NO2 (Nitrogen Dioxide): This reddish-brown gas comes from combustion processes, such as those in cars and power plants. High levels of NO2 can aggravate respiratory diseases, particularly asthma, leading to respiratory symptoms.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">SO2 (Sulfur Dioxide): Mainly produced from burning fossil fuels containing sulfur and from volcanic eruptions, SO2 is a gas that can affect the human respiratory system and make breathing difficult. It also contributes to the formation of acid rain.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">CO (Carbon Monoxide): This colorless, odorless gas is a product of incomplete combustion. High levels of CO in the air are especially dangerous because they can prevent oxygen from entering the cells and tissues.</p>', unsafe_allow_html=True)
    
    col1,col2,col3=st.columns(3,gap='small')
    show_nan = col1.checkbox('Show NaN distribution of the dataset')
    show_stati = col2.checkbox('Show statistical properties of the dataset')
    show_corr = col3.checkbox('Show correlation of the dataset')

    #Heatmap: see the NaN distribution
    if show_nan==True:
        plt.figure(figsize=(10, 4))
        sns.heatmap(site_data.isna().transpose(), cmap="magma")
        st.pyplot(plt)
        st.markdown('<p class="font_subtext1">Fig.1 The NaN distribution of the dataset</p>', unsafe_allow_html=True)

  
    #descriptive statistics
    if show_stati==True:
        df_statistics = site_data.describe()
        st.markdown('<p class="font_subsubtext1">Table 1. The descriptive statistics of the dataset </p>', unsafe_allow_html=True)
        st.write(df_statistics, height=800)
    

    #Heatmap: correlation
    if show_corr==True:
        correlation_matrix = site_data.corr(method='pearson')
        plt.figure(figsize=(22, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt)
        st.markdown('<p class="font_subtext1">Fig.2 Correlation diagram of the dataset</p>', unsafe_allow_html=True)

    st.markdown('<p class="font_text">Observations from Figure 1 and Table 1 reveal the presence of missing values within our dataset. To address this, we will employ a method that utilizes the "n nearest neighbors" technique for imputing these missing values. This approach will allow us to maintain the integrity and continuity of our dataset, ensuring a more robust and accurate analysis.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"></p>', unsafe_allow_html=True)
    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################

##################################################################################################################################################################

## Handle missing values by n nearest neighbors
handle_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3','TEMP','PRES','DEWP','WSPM']

imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(site_data[handle_columns])
imputed_site_data = pd.DataFrame(imputed_data, columns=handle_columns, index=site_data.index)
site_data.drop(columns=handle_columns, inplace=True)
site_data = site_data.join(imputed_site_data)


##################################################################################################################################################################
##Investigate the relationship of any two variables 
with tab2:
    st.markdown('<p class="font_subheader">Regplot: </p>', unsafe_allow_html=True)
    data_year = st.selectbox('Select the year:',(2014, 2015, 2016), index=0)

    site_data_yearly = site_data[site_data['year']==data_year]#only choose the data in 2016
    site_data_yearly.reset_index(drop=True, inplace=True)#Resetting the index

    # User selection
    columns = site_data_yearly.columns.tolist()
    col1,col2=st.columns(2,gap='small')
    x_axis = col1.selectbox("Select X variable:", options=columns,index=columns.index('PM2.5'))
    y_axis = col2.selectbox("Select Y variable:", options=columns,index=columns.index('PM10'))
    

#plotly regression
    if site_data_yearly[x_axis].dtype != 'object' and site_data_yearly[y_axis].dtype != 'object':
        fig = px.scatter(site_data_yearly, x=x_axis, y=y_axis, opacity=0.4, trendline='ols', 
            trendline_color_override='red'
            )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="font_subtext1">Fig.3 Relationship investigation of any two variables </p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="font_text">The chosen column has non-numerical items.</p>', unsafe_allow_html=True)

    
    st.markdown('<p class="font_text">The regplot showcased is a dynamic tool designed to analyze the interdependence between two selected variables over various years. By incorporating a feature to choose the year, the plot provides a tailored view of the correlations between different variables annually. Our analysis reveals a general trend: as concentrations of PM2.5, PM10, NO2, SO2, and CO escalate, there is a corresponding increase in the levels of the other pollutants. Notably, ozone (O3) is an exception to this pattern, indicating a unique behavior in contrast to the other measured substances. This nuanced insight underscores the complexity of air quality factors and their interactions across time.</p>', unsafe_allow_html=True)


    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################


##################################################################################################################################################################

# Combine year, month, day, and hour into a datetime column
site_data['datetime'] = pd.to_datetime(site_data[['year', 'month', 'day', 'hour']])
# Set datetime as the index
site_data.set_index('datetime', inplace=True)

# Seasonal averages (assuming Q1 starts in January)
seasonal_avg = site_data.resample('Q').mean()

# Monthly averages
monthly_avg = site_data.resample('M').mean()

# Weekly averages
weekly_avg = site_data.resample('W').mean()

# Daily averages
daily_avg = site_data.resample('D').mean()

# # Calculate the maximum values for each period
# seasonal_max = site_data.resample('Q').max()
# monthly_max = site_data.resample('M').max()
# weekly_max = site_data.resample('W').max()
# daily_max = site_data.resample('D').max()

# # Calculate the normalized averages by dividing by the maximum values
# normalized_seasonal_avg = seasonal_avg / seasonal_max
# normalized_monthly_avg = monthly_avg / monthly_max
# normalized_weekly_avg = weekly_avg / weekly_max
# normalized_daily_avg = daily_avg / daily_max

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']


##################################################################################################################################################################
##Find the air quality guidelines (AQG) limits that pose health risk
with tab3:

    st.markdown('<p class="font_text">The World Health Organization Global Air Quality Guidelines (AQG) provide a framework of threshold values and limits for critical air pollutants known to have adverse effects on health. These guidelines serve as a benchmark for protecting public health from the dangers of air pollution worldwide.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">Researchers and policymakers can utilize assumed AQG values as a reference point to compute and analyze the proportion of pollutant measurements that exceed these recommended thresholds. By doing so, they can assess the extent of air quality issues and identify periods or areas where pollutant concentrations pose significant health risks. This analysis is crucial for formulating strategies to improve air quality and safeguard public health, adapting the global guidelines to the specific context of each locality.</p>', unsafe_allow_html=True)

    
    st.markdown('<p class="font_text">Table 3.24 is the reference value used to set the assumed AQG in the below histplot: </p>', unsafe_allow_html=True)
    image1 = Image.open('AQG.png')
    st.image(image1, width=600)

    # User selection
    histplot_option = st.selectbox(
        'Select a pollutant variable for the histplot:',
        ('PM2.5', 'PM10','SO2', 'NO2', 'CO', 'O3'), index=0
    )
    
    plt.figure(figsize=(12, 8))
    plt.hist(daily_avg[histplot_option], bins=30, edgecolor='black')
    plt.axvline(x=np.mean(daily_avg[histplot_option]), linewidth=2, color='g',label = "Mean")

#Claculate the ratio that is larger than the assumed AQI 
    def cal_ratio(parameter, AQG):
        plt.axvline(x=AQG, linewidth=2, color='r',linestyle='--', label = "Assumed AQG")
        count = 0
        for i in range(len(daily_avg[histplot_option])):
            if daily_avg[histplot_option][i] > AQG:
                count += 1
        r = count/len(daily_avg[histplot_option])
        return r

    if histplot_option == 'PM2.5':
        r1 = cal_ratio(histplot_option, 25)
        st.write('For', histplot_option, ', the ratio larger than the assumed AQG is', r1)

    if histplot_option == 'PM10':
        r1 = cal_ratio(histplot_option, 45)
        st.write('For', histplot_option, ', the ratio larger than the assumed AQG is', r1)

    if histplot_option == 'SO2':
        r1 = cal_ratio(histplot_option, 40)
        st.write('For', histplot_option, ', the ratio larger than the assumed AQG is', r1)

    if histplot_option == 'NO2':
        r1 = cal_ratio(histplot_option, 25)
        st.write('For', histplot_option, ', the ratio larger than the assumed AQG is', r1)

    if histplot_option == 'CO':
        r1 = cal_ratio(histplot_option, 4000)
        st.write('For', histplot_option, ', the ratio larger than the assumed AQG is', r1)

    if histplot_option == 'O3':
        r1 = cal_ratio(histplot_option, 100)
        st.write('For', histplot_option, ', the ratio larger than the assumed AQG is', r1)


    plt.xlabel(f" {histplot_option} ")
    plt.ylabel("count")
    plt.legend(loc=0)
    plt.grid()
    st.pyplot(plt)
    st.markdown('<p class="font_subtext1">Fig.4 A histplot to show where are the average and AQG</p>', unsafe_allow_html=True)


    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################


##################################################################################################################################################################
#calculate the AQI
with tab4:
    st.markdown('<p class="font_text">The Air Quality Index (AQI) is calculated based on the highest concentration of the air pollutants, typically including PM2.5, PM10, ozone (O3), nitrogen dioxide (NO2), sulfur dioxide (SO2), and carbon monoxide (CO). </p>', unsafe_allow_html=True)

    st.markdown('<p class="font_text">Here we define:</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Excelent when the highest concentration is less than 50.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Good when the highest concentration is less than 100.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Slight when the highest concentration is less than 150.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Light when the highest concentration is less than 200.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Moderate when the highest concentration is less than 250.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Heavy when the highest concentration is less than 300.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text"> -- Severe when the highest concentration is more than 300.</p>', unsafe_allow_html=True)


    site_data["CO"] = site_data["CO"]/1000 #change the unit ug/m^3 to mg/m^3
    def calculate_aqi(row):
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        max_value = row[pollutants].max()
    
        if max_value <= 50:
            return 'Excellent'
        elif max_value <= 100:
            return 'Good'
        elif max_value <= 150:
            return 'Slight'
        elif max_value <= 200:
            return 'Light'
        elif max_value <= 250:
            return 'Moderate'
        elif max_value <= 300:
            return 'Heavy'
        else:
            return 'Severe'
    
    site_data['AQI'] = site_data.apply(calculate_aqi, axis=1)

    plt.figure(figsize=(12, 8))
    plt.hist(site_data['AQI'], bins=7, edgecolor='black')
    plt.xlabel("AQI")
    plt.ylabel("count")
    plt.grid()
    plt.legend(loc=0)
    st.pyplot(plt)
    st.markdown('<p class="font_subtext1">Fig.5 AQI distribution in whole year (one count per hour)</p>', unsafe_allow_html=True)

    st.markdown('<p class="font_text">In Fig.5, it shows the air quality is good under this definition as the good histplot is the highest counting number.</p>', unsafe_allow_html=True)

    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################


##################################################################################################################################################################
#1. seasonal Analyze
with tab5:
    
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    winter_months = [12, 1, 2]

    df_spring = site_data[site_data['month'].isin(spring_months)]# Filter the data to include only the spring months
    spring_averages_by_hour = df_spring.groupby('hour').mean()# Now, group by 'hour' to get the average spring values for each hour

    df_summer = site_data[site_data['month'].isin(summer_months)]
    summer_averages_by_hour = df_summer.groupby('hour').mean()

    df_fall = site_data[site_data['month'].isin(fall_months)]
    fall_averages_by_hour = df_fall.groupby('hour').mean()

    df_winter = site_data[site_data['month'].isin(winter_months)]
    winter_averages_by_hour = df_winter.groupby('hour').mean()
    
    # fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), constrained_layout=True)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), tight_layout=True)
    axes = axes.flatten()# Flatten the axes array for easy iteration
    time = np.arange(24)

    # Plot each pollutant in its own subplot
    for i, pollutant in enumerate(pollutants):
        axes[i].plot(time, spring_averages_by_hour[pollutant], label='Spring')
        axes[i].scatter(time, spring_averages_by_hour[pollutant])

        axes[i].plot(time, summer_averages_by_hour[pollutant], label='Summer')
        axes[i].scatter(time, summer_averages_by_hour[pollutant])

        axes[i].plot(time, fall_averages_by_hour[pollutant], label='Fall')
        axes[i].scatter(time, fall_averages_by_hour[pollutant])

        axes[i].plot(time, winter_averages_by_hour[pollutant], label='Winter')
        axes[i].scatter(time, winter_averages_by_hour[pollutant])

        axes[i].set_title(f'Seasonal Average of {pollutant}')
        axes[i].set_xlabel('time')
        axes[i].set_ylabel('Concentration')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xticklabels(axes[i].get_xticklabels()) #Rotate x-axis tick labels to vertical
    plt.tight_layout()
    st.pyplot(plt)
    st.markdown('<p class="font_subtext1">Fig.6 The average seasonal concentration of pollutants with varying time from 0 am to 11 pm(23). We define Spring(3, 4, 5), Summer(6, 7, 8), Fall(9, 10, 11), and Winter(12, 1, 2).</p>', unsafe_allow_html=True)

    st.markdown('<p class="font_text">Fig.6 presents a comparative analysis of the average seasonal concentrations for six pollutants: PM2.5, PM10, SO2, NO2, CO, and O3. The x-axis represents time across a 24-hour period, denoted from 0 to 23 hours, corresponding to the hours of the day from midnight to 11 PM. The y-axis measures pollutant concentration levels. </p>', unsafe_allow_html=True)

    st.markdown('<p class="font_text">PM2.5 and PM10: Both show similar patterns in Fig.6, with concentrations peaking during the late evening and early morning hours. This could be due to lower atmospheric mixing during these times, allowing pollutants to accumulate near the ground. Winter shows the highest levels, possibly due to increased heating activities and temperature inversions that trap pollutants close to the surface. PM10 in Spring shows the highest value than other seasons, which may be caused by the pollen and catkins.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">SO2 and NO2: These pollutants also show higher concentrations during the night and early morning hours, with a noticeable dip during midday. This pattern might result from reduced traffic and industrial activities at night, with a resurgence in the morning. The winter season again shows higher concentrations, potentially due to the same reasons as PM2.5 and PM10.</p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">CO: Exhibits a pronounced peak during the winter evenings, which might be attributed to increased combustion from heating and vehicles in colder weather, coupled with stagnant air that fails to disperse the pollutants. </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text">O3: Unlike the other pollutants, ozone has its peak during the daytime, especially in the summer, which is consistent with the photochemical generation of ozone from precursor emissions (like NOx and VOCs) in the presence of sunlight. Lower levels in winter could be due to the reduced intensity of sunlight and shorter days. </p>', unsafe_allow_html=True)

    # 2. Monthly analysis
    # col1,col2,col3=st.columns(3,gap='small')
    show_monthly_analysis = st.checkbox('Show average monthly concentration of pollutants with varying month')
    
    if show_monthly_analysis==True:
        plt.figure(figsize=(8, 8))
        monthly_avg_filtered = monthly_avg['2014':'2016']# Filter the DataFrame for the years 2013 to 2016
        grouped = monthly_avg_filtered.groupby(monthly_avg_filtered.index.month)# Group the data by year

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), constrained_layout=True)
        axes = axes.flatten()# Flatten the axes array for easy iteration
        month = np.arange(len(monthly_avg_filtered['PM2.5']))

        # Plot each pollutant in its own subplot
        for i, pollutant in enumerate(pollutants):
            axes[i].plot(month, monthly_avg_filtered[pollutant], color='g')
            axes[i].scatter(month, monthly_avg_filtered[pollutant], color='g')
            axes[i].set_title(f'Monthly Average of {pollutant}')
            axes[i].set_xlabel('Months')
            axes[i].set_ylabel('Concentration')
            axes[i].grid(True)
            axes[i].set_xticklabels(axes[i].get_xticklabels()) #Rotate x-axis tick labels to vertical
        plt.tight_layout()
        st.pyplot(plt)
        st.markdown('<p class="font_subtext1">Fig.7 The average monthly concentration of pollutants with varying month start from Janurary in 2014-2016.</p>', unsafe_allow_html=True)

        st.markdown('<p class="font_text">The most summary get from the fig.6 can be observed from fig.7, too. Fig.7 illustrates the average monthly concentrations of key pollutants over the months, beginning from January 2014 through to the end of 2016. This graph provides a visual representation of the fluctuating levels of various air contaminants and helps to identify seasonal trends and potential anomalies within this three-year span.</p>', unsafe_allow_html=True)
     
    
    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################



##################################################################################################################################################################
##Time Series Analysis
with tab6:
    st.markdown('<p class="font_text">The air quality data is related to environmental measurements over a period of time. This kind of data is often used in time series analysis(TSA) to forecast future values based on historical trends and patterns. Here, we use the one-step-ahead-forecasting. In time series analysis, it refers to the prediction of a future value in the series at a specific number of time steps ahead from the current point. </p>', unsafe_allow_html=True)

    pollutants_tuple = ('PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3')
    default_pollutant_index = pollutants_tuple.index('SO2')
    pollutant = st.selectbox('Select a pollutant:', pollutants_tuple, index=default_pollutant_index)

    fig, ax = plt.subplots(1, 1, figsize = (15,4))
    sns.scatterplot(data=daily_avg, x="datetime", y=pollutant, label="The whole actual data", ax=ax)

    # Setting the title and labels
    _ = ax.set(xlabel="Date", ylabel=f"{pollutant} concentration")
    
    st.pyplot(fig)
    st.markdown('<p class="font_subtext1">Fig.8 The daily average pollutant concentration with date.</p>', unsafe_allow_html=True)


    #N-Step-Ahead-Forecasting
    total_number = len(daily_avg[pollutant])
    step_size = max(int(total_number / 200), 1)  # Ensure step size is at least 1
    temps = daily_avg[pollutant][::step_size].to_numpy()
    chosen_sample_number = len(temps)
    st.write('We choose', chosen_sample_number, 'data from', total_number, 'actual daily average data to do the daily prediction.')
    full_temps = copy(temps)
    temps=temps[0:170]

    def organize_dataset(signal, N=1):
        X, y = [], []
        for i in range(len(signal) - N):
            a = signal[i:(i + N)]
            X.append(a)
            y.append(signal[i + N])
        return np.array(X), np.array(y)

    def predict_next_value(input_vector, a):
        return np.dot(input_vector, a)

    N = st.slider('Input the number of the first data points', 1, 100, 50)#The default value is the last number
    
    X, y = organize_dataset(temps, N)

    update = np.linalg.pinv(X)@y
    
    # next value
    last_values = temps[-N:]
    next_value = predict_next_value(last_values, update)

    plt.figure(figsize=(15,5))
    old_signal = temps.copy()
    new_signal = np.append(temps, next_value)
    steps = 40
    for _ in range(steps):
        last_values = new_signal[-N:]
        next_value = predict_next_value(last_values, update)
        new_signal = np.append(new_signal, next_value)

    plt.title(f"Prediction vs Actual Data for A-Step-Ahead-Forecasting with N = {N}")
    plt.plot(temps, label='training')
    plt.plot(full_temps, 'o', alpha=0.4, label='truth')
    plt.plot(new_signal, '-^', alpha=0.4, label='forecast')
    plt.xlabel('time')
    plt.ylabel(f'{pollutant} Concentration')
    plt.legend()
    plt.grid(alpha=0.2)

    st.pyplot(plt)

    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################


##################################################################################################################################################################
## Feature(s) selection
with tab7:
    st.markdown('<p class="font_text">Feature selection is a critical process in model building, involves identifying and selecting a subset of input variables that contribute the most to the prediction of the target variable. This technique is essential in machine learning and data modeling, as it helps in enhancing the model performance by eliminating redundant or irrelevant data, reducing overfitting, and improving computational efficiency. </p>', unsafe_allow_html=True)
    
    st.markdown('<p class="font_text">In our analysis, we employ a diverse array of models, namely Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, Neural Net, and AdaBoost. Each of these models is fine-tuned with one adjustable parameter, with the exception of the AdaBoost model. We meticulously generate and display figures for each models predictions, juxtaposing these against actual data for a comprehensive comparison. Furthermore, we rigorously evaluate each models performance by calculating its score and the Mean Squared Error (MSE), providing a quantitative measure of their predictive accuracy and reliability. </p>', unsafe_allow_html=True)

    # data_names = ['seasonal_avg','monthly_avg','daily_avg', 'site_data']
    # data_list = [seasonal_avg, monthly_avg, daily_avg, site_data]

    # default_index = data_names.index('daily_avg')

    # data_name = st.selectbox('Select a data you are interested', data_names, index=default_index)
    # data_index = data_names.index(data_name)
    # target_data = data_list[data_index]
    target_data = daily_avg

    # Create a slider widget for test_fraction
    test_fraction = st.slider('Select a value for the test_fraction:', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    # User selection
    col1 , col2 = st.columns(2,gap='small')
    feature_variables = col1.multiselect('Select feature(s) for machine learning:', ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3','TEMP','PRES','DEWP','WSPM'], default = 'TEMP')
    target_variable = col2.selectbox('Select target feature for the machine learning:', ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'], index=5)

    # Define hyperparameter sliders before the loop
    col1, col2, col3, col4, col5, col6 = st.columns(6, gap='small')
    knn = col1.slider('Select the nearest neighbors for KNeighborsRegressor:', min_value=1, max_value=10, value=3, step=1)
    cc = col2.slider('Select the regularization parameter for Linear SVM:', min_value=0.01, max_value=10.0, value=0.025, step=0.02)
    gg = col3.selectbox('Select gamma for RBF SVM:', [0.0001, 0.001, 0.01, 0.1, 10], index=3)
    depth = col4.slider('Select the maximum depth for DecisionTreeRegressor:', min_value=1, max_value=10, value=5, step=1)
    nn = col5.slider('Select the estimators for RandomForestRegressor:', min_value=10, max_value=100, value=10, step=10)
    alph = col6.number_input('Input a non-negative value for alpha of MLPRegressor: ', value=0.1, format='%f')
   
    # Updated names for regression
    model_names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
    ]

    # Updated classifiers for regression
    classifiers_list = [
        KNeighborsRegressor(knn),
        SVR(kernel="linear", C=cc),
        SVR(gamma=gg, C=1),
        DecisionTreeRegressor(max_depth=depth, random_state=42),
        RandomForestRegressor(max_depth=5, n_estimators=nn, max_features=1, random_state=42),
        MLPRegressor(alpha=alph, max_iter=1000, random_state=42),
        AdaBoostRegressor(random_state=42),
    ]


    X = target_data[feature_variables]
    y = target_data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=42)

    name = st.selectbox('Select the model', model_names, index=0)
    model_index = model_names.index(name)
    select_model = classifiers_list[model_index]

    select_model.fit(X_train, y_train)
    y_pred = select_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score = select_model.score(X_test, y_test)
    st.write('Model Name:', name)
    st.write('Model Score:', score)
    st.write('Mean Squared Error:', mse)  
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.scatter(X_test.index, y_test, color='blue', label='Actual Data', alpha=0.6)
    plt.scatter(X_test.index, y_pred, color='red', label='Predicted Data', alpha=0.3)
    plt.title(f'Prediction vs Actual Data for {name}')
    plt.xlabel('Date')
    plt.ylabel(f'{target_variable} Concentration')
    plt.legend()

    st.pyplot(plt)

    # # iterate over classifiers
    # for name, clf in zip(model_names, classifiers_list):
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     mse = mean_squared_error(y_test, y_pred)
    #     score = clf.score(X_test, y_test)
    #     st.write('Model Name:', name)
    #     st.write('Model Score:', score)
    #     st.write('Mean Squared Error:', mse)  
        
    #     # Plotting
    #     plt.figure(figsize=(15, 6))
    #     plt.scatter(X_test.index, y_test, color='blue', label='Actual Data', alpha=0.6)
    #     plt.scatter(X_test.index, y_pred, color='red', label='Predicted Data', alpha=0.3)
    #     plt.title(f'Prediction vs Actual Data for {name}')
    #     plt.xlabel('Date')
    #     plt.ylabel(f'{target_variable} Concentration')
    #     plt.legend()

    #     st.pyplot(plt)

    ####################################################################################################################################################################

    st.markdown('<p class="font_header">Reference </p>', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[1] https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[2] https://www.who.int/publications/i/item/9789240034228.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[3] Xie S, Yu T, Zhang Y, et al. Characteristics of PM10, SO2, NOx and O3 in ambient air during the dust storm period in Beijing[J]. Science of the Total Environment, 2005, 345(1-3): 153-164.', unsafe_allow_html=True)
    st.markdown('<p class="font_text1">[4] https://github.com/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set/blob/main/notebooks/analyticis.ipynb.', unsafe_allow_html=True)

    ##################################################################################################################################################################
