import streamlit as st
import streamlit.ReportThread as ReportThread
from streamlit.server.Server import Server
import pandas as pd
import pickle

# essential libraries
import math
import random
import string
from datetime import timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import warnings

warnings.filterwarnings('ignore')


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    if st.sidebar.button("Download new data"):
        getDataToCsv()
        st.caching.clear_cache()
    full_table, full_grouped, day_wise, country_wise = loadData()

    lastUpdate = checkLast(full_table)
    st.sidebar.text("Last update: " + lastUpdate)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("",
                                    ["Introduction",
                                     "View Graphs"])
                                     #"Show the source code"])

    if app_mode == "Introduction":
        # Render the readme as markdown using st.markdown.
        readme_text = st.markdown(get_file_content_as_string("instructions.md"))
        dicti = pickle.load(open("settings.p", "rb"))
        dicti[get_session_id()] = 0
        print(len(dicti))
        pickle.dump(dicti, open("settings.p", "wb"))
        st.sidebar.success('To continue select "View Graphs".')
    elif app_mode == "Show the source code":
        st.code(get_file_content_as_string("app.py"))
    elif app_mode == "Show the source code":
        showMode()
    elif app_mode == "View Graphs":
        html = """
                <style>
                /* Remove horizontal scroll */
                .main .block-container > div:first-of-type {
                  overflow-x: hidden !important;
                  width: 730px !important;
                }
                
                .main .block-container {
                  width: 730px !important;
                }

                .fullScreenFrame > div {
                  width: auto !important;
                }
                /* 1nd button */
                .main .element-container:nth-child(3) {
                    left: 30px;
                    top: 0px;
                }

                /* Main Title*/
                .main .element-container:nth-child(4) {
                  top: -75.5px;
                  left: 145px;
                }

                /* 2nd button */
                .main .element-container:nth-child(5) {
                  top: -121px;
                  left: 580px;
                }
                /* Selector */
                .main .element-container:nth-child(6) {
                  top: -95.5px;
                }
                /* First Graph */
                .main .element-container:nth-child(8) {
                  top: -105.5px;
                }
                /* Second Graph */
                .main .element-container:nth-child(9) {
                  top: -105.5px;
                }
                /* Third Graph */
                .main .element-container:nth-child(10) {
                  top: -105.5px;
                }
                
                .main .element-container:nth-child(11) { top: -105.5px; }
                .main .element-container:nth-child(12) { top: -105.5px; }
                .main .element-container:nth-child(13) { top: -105.5px; }
                .main .element-container:nth-child(14) { top: -105.5px; }
                .main .element-container:nth-child(15) { top: -105.5px; }
                .main .element-container:nth-child(16) { top: -105.5px; }
                .main .element-container:nth-child(17) { top: -105.5px; }
                .main .element-container:nth-child(18) { top: -105.5px; }
                .main .element-container:nth-child(19) { top: -105.5px; }
                .main .element-container:nth-child(20) { top: -105.5px; }
                .main .element-container:nth-child(21) { top: -105.5px; }
                .main .element-container:nth-child(22) { top: -105.5px; }
                .main .element-container:nth-child(23) { top: -105.5px; }
                .main .element-container:nth-child(24) { top: -105.5px; }
                .main .element-container:nth-child(25) { top: -105.5px; }
                .main .element-container:nth-child(26) { top: -105.5px; }
                .main .element-container:nth-child(27) { top: -105.5px; }
                .main .element-container:nth-child(28) { top: -105.5px; }
                .main .element-container:nth-child(29) { top: -105.5px; }
                .main .element-container:nth-child(30) { top: -105.5px; }
                .main .element-container:nth-child(31) { top: -105.5px; }
                .main .element-container:nth-child(32) { top: -105.5px; }
                .main .element-container:nth-child(33) { top: -105.5px; }
                .main .element-container:nth-child(34) { top: -105.5px; }
                .main .element-container:nth-child(35) { top: -105.5px; }
                .main .element-container:nth-child(36) { top: -105.5px; }
                .main .element-container:nth-child(37) { top: -105.5px; }
                .main .element-container:nth-child(38) { top: -105.5px; }
                .main .element-container:nth-child(39) { top: -105.5px; }
                .main .element-container:nth-child(40) { top: -105.5px; }
                .main .element-container:nth-child(41) { top: -105.5px; }
                .main .element-container:nth-child(42) { top: -105.5px; }
                .main .element-container:nth-child(43) { top: -105.5px; }
                .main .element-container:nth-child(44) { top: -105.5px; }
                .main .element-container:nth-child(45) { top: -105.5px; }
                .main .element-container:nth-child(46) { top: -105.5px; }
                .main .element-container:nth-child(47) { top: -105.5px; }
                .main .element-container:nth-child(48) { top: -105.5px; }
                .main .element-container:nth-child(49) { top: -105.5px; }
                .main .element-container:nth-child(50) { top: -105.5px; }
                .main .element-container:nth-child(51) { top: -105.5px; }
                .main .element-container:nth-child(52) { top: -105.5px; }
                .main .element-container:nth-child(53) { top: -105.5px; }
                .main .element-container:nth-child(54) { top: -105.5px; }
                .main .element-container:nth-child(55) { top: -105.5px; }
                .main .element-container:nth-child(56) { top: -105.5px; }
                .main .element-container:nth-child(57) { top: -105.5px; }
                .main .element-container:nth-child(58) { top: -105.5px; }
                .main .element-container:nth-child(59) { top: -105.5px; }
                .main .element-container:nth-child(60) { top: -105.5px; }
                .main .element-container:nth-child(61) { top: -105.5px; }
                .main .element-container:nth-child(62) { top: -105.5px; }
                .main .element-container:nth-child(63) { top: -105.5px; }
                .main .element-container:nth-child(64) { top: -105.5px; }
                .main .element-container:nth-child(65) { top: -105.5px; }
                .main .element-container:nth-child(66) { top: -105.5px; }
                .main .element-container:nth-child(67) { top: -105.5px; }
                .main .element-container:nth-child(68) { top: -105.5px; }

                </style>
                """
        st.markdown(html, unsafe_allow_html=True)
        dicti = pickle.load(open("settings.p", "rb"))
        mode = dicti[get_session_id()]
        button = 0
        if st.button("Previous"):
            button = -1
        st.title("Covid Data Visualization")
        if st.button("Next"):
            button = 1
        mode += button
        if mode < 0:
            mode = 0
        if mode > 19:
            mode = 19

        selector = graphSelectorUI(mode)
        if getIndex(selector) != mode:
            mode = getIndex(selector)

        dicti[get_session_id()] = mode
        pickle.dump(dicti, open("settings.p", "wb"))

        for i in range(mode * 3):
            st.empty()

        for content in showMode(mode, full_table, full_grouped, day_wise, country_wise):
            st.plotly_chart(content, use_container_width=True)

        readme_text = st.markdown(get_file_content_as_string("footer.md"), unsafe_allow_html=True)


def graphSelectorUI(index=0):
    return st.selectbox("Choose:",
                        ["1. World Graphs", "2. World Graph Over Time", "3. Case Composition", "4. Bar Graph",
                         "5. Cases Over Time", "6. Summary", "7. Top 15 Countries", "8. Scatter Graph",
                         "9. Countries Over Time", "10. Countries Over Time Stacked",
                         "11. Countries From Case N", "12. Countries By Percentage",
                         "13. New Cases By Country", "14. Active Cases By Country",
                         "15. Epidemic Time Span By Country", "16. Countries Histograms",
                         "17. Region Analysis", "18. Weekly Analysis", "19. Monthly Analysis",
                         "20. Comparison against similar viruses"], index=index)


def getIndex(selector):
    if selector == "1. World Graphs":
        return 0
    elif selector == "2. World Graph Over Time":
        return 1
    elif selector == "3. Case Composition":
        return 2
    elif selector == "4. Bar Graph":
        return 3
    elif selector == "5. Cases Over Time":
        return 4
    elif selector == "6. Summary":
        return 5
    elif selector == "7. Top 15 Countries":
        return 6
    elif selector == "8. Scatter Graph":
        return 7
    elif selector == "9. Countries Over Time":
        return 8
    elif selector == "10. Countries Over Time Stacked":
        return 9
    elif selector == "11. Countries From Case N":
        return 10
    elif selector == "12. Countries By Percentage":
        return 11
    elif selector == "13. New Cases By Country":
        return 12
    elif selector == "14. Active Cases By Country":
        return 13
    elif selector == "15. Epidemic Time Span By Country":
        return 14
    elif selector == "16. Countries Histograms":
        return 15
    elif selector == "17. Region Analysis":
        return 16
    elif selector == "18. Weekly Analysis":
        return 17
    elif selector == "19. Monthly Analysis":
        return 18
    elif selector == "20. Comparison against similar viruses":
        return 19
    return 0


def showMode(mode, full_table, full_grouped, day_wise, country_wise):
    if mode <= 0:
        return showWorldMap(country_wise)
    elif mode == 1:
        return worldTimeGraph(full_grouped)
    elif mode == 2:
        return caseComposition(full_table)
    elif mode == 3:
        return showBarGraph(full_table)
    elif mode == 4:
        return showTimeSeries(full_table)
    elif mode == 5:
        loga = st.sidebar.checkbox("Logarithmic Scales (On first row)")
        return [casesOverTime(day_wise, loga), casesOverTime2(day_wise), casesOverTime3(day_wise)]
    elif mode == 6:
        selected = st.sidebar.multiselect("Compare with:", getCountryList(full_table, full_grouped))
        return top15(country_wise, selected)
    elif mode == 7:
        slider = st.sidebar.slider("Number of countries", 1, 100, 20, 1)
        return showScatter(country_wise, slider)
    elif mode == 8:
        st.sidebar.markdown("Tip: Click on countries legends to show/hide them")
        return dateGraph2(full_grouped)
    elif mode == 9:
        return dateGraph(full_grouped)
    elif mode == 10:
        slider2 = st.sidebar.slider("Change N", 1, 10000, 100, 1)
        loga2 = st.sidebar.checkbox("Logarithmic Scale")
        return fromCaseN(full_grouped, full_table, slider2, loga2)
    elif mode == 11:
        return percentageCases(full_grouped, day_wise)
    elif mode == 12:
        radio = st.sidebar.radio("Sort:", ["Alphabetically", "Most Cases", "Most Active Cases"])
        return newCasesTime(full_grouped, radio)
    elif mode == 13:
        radio2 = st.sidebar.radio("Sort:", ["Alphabetically", "Most Cases", "Most Active Cases"])
        return activeCasesTime(full_grouped, radio2)
    elif mode == 14:
        radio3 = st.sidebar.radio("Sort:", ["Alphabetically", "Longest Duration", "Shortest Duration"], 1)
        return epidemicSpan(full_table, radio3)
    elif mode == 15:
        loga3 = st.sidebar.checkbox("Logarithmic Scales")
        slider3 = st.sidebar.slider("Countries with more than N total cases", 10000, 100000, 50000, 10000)
        return showCountryWise(full_table, full_grouped, loga3, slider3)
    elif mode == 16:
        perMillion = st.sidebar.checkbox("Display per million people (Scatter Graph)")

        st.sidebar.subheader("Regions")
        st.sidebar.markdown("AFRO - African Region")
        st.sidebar.markdown("EMRO - Eastern Mediterranean Region")
        st.sidebar.markdown("EURO - European Region")
        st.sidebar.markdown("PAHO - Pan American Health Organization")
        st.sidebar.markdown("WPRO - Western Pacific Region")
        st.sidebar.markdown("SEARO - South-East Asia Region")

        figs = whos(country_wise, full_grouped, perMillion)
        st.write(figs[0])
        return [figs[1], figs[2]]
    elif mode == 17:
        loga4 = st.sidebar.checkbox("Logarithmic Scales")
        return weeklyStatistics(full_grouped, loga4)
    elif mode == 18:
        loga5 = st.sidebar.checkbox("Logarithmic Scales")
        return monthlyStatistics(full_grouped, loga5)
    elif mode >= 19:
        return similarComparisons(full_table)


@st.cache(show_spinner=True, allow_output_mutation=True)
def getCountryList(full_table, full_grouped):
    temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']
    temp = temp.sum().diff().reset_index()

    mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

    temp.loc[mask, 'Confirmed'] = np.nan
    temp.loc[mask, 'Deaths'] = np.nan

    gt_10000 = full_grouped[full_grouped['Confirmed'] > 1000]['Country/Region'].unique()
    temp = temp[temp['Country/Region'].isin(gt_10000)]

    # countries = ['China', 'Iran', 'South Korea', 'Italy', 'France', 'Germany', 'Italy', 'Spain', 'US']
    countries = temp['Country/Region'].unique()
    return countries


@st.cache(show_spinner=True, allow_output_mutation=True)
def monthlyStatistics(full_grouped, loga):
    full_grouped['Month'] = pd.DatetimeIndex(full_grouped['Date']).month
    full_grouped['Week No.'] = full_grouped['Date'].dt.strftime('%U')
    month_wise = full_grouped.groupby('Month')[
        'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'].sum().reset_index()

    # ---------------

    # Monthly statistics
    # ================

    fig_c = px.bar(month_wise, x="Month", y="Confirmed", color_discrete_sequence=['#000000'])
    fig_d = px.bar(month_wise, x="Month", y="Deaths", color_discrete_sequence=['#ff677d'])
    fig_r = px.bar(month_wise, x="Month", y="Recovered", color_discrete_sequence=['#649d66'])

    fig_dc = px.bar(month_wise, x="Month", y="New cases", color_discrete_sequence=['#323232'])
    fig_dd = px.bar(month_wise, x="Month", y="New deaths", color_discrete_sequence=['#cd6684'])
    fig_dr = px.bar(month_wise, x="Month", y="New recovered", color_discrete_sequence=['#16817a'])

    fig = make_subplots(rows=3, cols=2, shared_xaxes=False,
                        horizontal_spacing=0.14, vertical_spacing=0.1,
                        subplot_titles=('Confirmed cases', 'New cases',
                                        'Deaths reported', 'New deaths',
                                        'Cured', 'New cured'))

    fig.add_trace(fig_c['data'][0], row=1, col=1)
    fig.add_trace(fig_dc['data'][0], row=1, col=2)

    fig.add_trace(fig_d['data'][0], row=2, col=1)
    fig.add_trace(fig_dd['data'][0], row=2, col=2)

    fig.add_trace(fig_r['data'][0], row=3, col=1)
    fig.add_trace(fig_dr['data'][0], row=3, col=2)

    if loga:
        fig.update_yaxes(type="log")
    fig.update_layout(height=1400, title_text="Monthly")

    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def weeklyStatistics(full_grouped, loga):
    full_grouped['Month'] = pd.DatetimeIndex(full_grouped['Date']).month
    full_grouped['Week No.'] = full_grouped['Date'].dt.strftime('%U')
    week_wise = full_grouped.groupby('Week No.')[
        'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered'].sum().reset_index()
    # Weekly statistics
    # ================

    fig_c = px.bar(week_wise, x="Week No.", y="Confirmed", color_discrete_sequence=['#000000'])
    fig_d = px.bar(week_wise, x="Week No.", y="Deaths", color_discrete_sequence=['#ff677d'])
    fig_r = px.bar(week_wise, x="Week No.", y="Recovered", color_discrete_sequence=['#649d66'])

    fig_dc = px.bar(week_wise, x="Week No.", y="New cases", color_discrete_sequence=['#323232'])
    fig_dd = px.bar(week_wise, x="Week No.", y="New deaths", color_discrete_sequence=['#cd6684'])
    fig_dr = px.bar(week_wise, x="Week No.", y="New recovered", color_discrete_sequence=['#16817a'])

    fig = make_subplots(rows=3, cols=2, shared_xaxes=False,
                        horizontal_spacing=0.14, vertical_spacing=0.1,
                        subplot_titles=('Confirmed cases', 'New cases',
                                        'Deaths reported', 'New deaths',
                                        'Cured', 'New cured'))

    fig.add_trace(fig_c['data'][0], row=1, col=1)
    fig.add_trace(fig_dc['data'][0], row=1, col=2)

    fig.add_trace(fig_d['data'][0], row=2, col=1)
    fig.add_trace(fig_dd['data'][0], row=2, col=2)

    fig.add_trace(fig_r['data'][0], row=3, col=1)
    fig.add_trace(fig_dr['data'][0], row=3, col=2)

    if loga:
        fig.update_yaxes(type="log")
    fig.update_layout(height=1400, title_text="Weekly")

    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def whos(country_wise, full_grouped, per_million):
    who = country_wise.groupby('WHO Region')['Confirmed', 'Deaths', 'Recovered', 'Active',
                                             'New cases', 'Population', 'Confirmed last week'].sum().reset_index()
    who['Fatality Rate'] = (who['Deaths'] / who['Confirmed']) * 100
    who['Recovery Rate'] = (who['Recovered'] / who['Confirmed']) * 100
    who['Cases / Million'] = (who['Confirmed'] / who['Population']) * 10 ** 6

    # ------------------------------------------------------

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('WHO Region Wise', fontsize=16)
    axes = axes.flatten()

    cols = ['Confirmed', 'Active', 'Deaths', 'Recovered',
            'New cases', 'Confirmed last week',
            'Fatality Rate', 'Recovery Rate', 'Cases / Million']

    for ind, col in enumerate(cols):
        sns.barplot(data=who.sort_values(col, ascending=False),
                    x=col, y='WHO Region', hue='WHO Region', dodge=False,
                    hue_order=who['WHO Region'],
                    palette='Dark2', ax=axes[ind])
        axes[ind].set_title(col)
        axes[ind].set_xlabel('')
        axes[ind].set_ylabel('')

    # -------------
    xName = 'Confirmed'
    yName = 'Deaths'
    if per_million:
        xName = "Cases / Million People"
        yName = "Deaths / Million People"


    fig1 = px.scatter(country_wise, x=xName, y=yName, color='WHO Region',
                      height=700, hover_name='Country/Region', log_x=True, log_y=True,
                      title='WHO Region wise',
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    fig1.update_traces(textposition='top center')
    # fig.update_layout(showlegend=False)
    # fig.update_layout(xaxis_rangeslider_visible=True)

    # -------------

    who_g = full_grouped.groupby(['WHO Region', 'Date'])['Confirmed', 'Deaths', 'Recovered',
                                                         'Active', 'New cases', 'New deaths'].sum().reset_index()
    fig2 = px.bar(who_g, x="Date", y="New cases", color='WHO Region',
                  height=600, title='New cases',
                  color_discrete_sequence=px.colors.qualitative.Vivid)

    return [fig, fig1, fig2]


@st.cache(show_spinner=True, allow_output_mutation=True)
def similarComparisons(full_table):
    full_latest = full_table[full_table['Date'] == max(full_table['Date'])]
    epidemics = pd.DataFrame({
        'epidemic': ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],
        'start_year': [2019, 2003, 2014, 2012, 2009],
        'end_year': [2020, 2004, 2016, 2017, 2010],
        'confirmed': [full_latest['Confirmed'].sum(), 8096, 28646, 2494, 6724149],
        'deaths': [full_latest['Deaths'].sum(), 774, 11323, 858, 19654]
    })

    epidemics['mortality'] = round((epidemics['deaths'] / epidemics['confirmed']) * 100, 2)

    temp = epidemics.melt(id_vars='epidemic', value_vars=['confirmed', 'deaths', 'mortality'],
                          var_name='Case', value_name='Value')

    fig = px.bar(temp, x="epidemic", y="Value", color='epidemic', text='Value', facet_col="Case",
                 color_discrete_sequence=px.colors.qualitative.Bold, height=400)
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(showticklabels=False)
    fig.layout.yaxis2.update(matches=None)
    fig.layout.yaxis3.update(matches=None)
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def showCountryWise(full_table, full_grouped, loga, slide):
    temp = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']
    temp = temp.sum().diff().reset_index()

    mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

    temp.loc[mask, 'Confirmed'] = np.nan
    temp.loc[mask, 'Deaths'] = np.nan

    gt_10000 = full_grouped[full_grouped['Confirmed'] > slide]['Country/Region'].unique()
    temp = temp[temp['Country/Region'].isin(gt_10000)]

    # countries = ['China', 'Iran', 'South Korea', 'Italy', 'France', 'Germany', 'Italy', 'Spain', 'US']
    countries = temp['Country/Region'].unique()

    n_cols = 3
    n_rows = math.ceil(len(countries) / n_cols)

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False, subplot_titles=countries)

    for ind, country in enumerate(countries):
        row = int((ind / n_cols) + 1)
        col = int((ind % n_cols) + 1)
        fig.add_trace(go.Bar(x=temp['Date'], y=temp.loc[temp['Country/Region'] == country, 'Confirmed'], name=country),
                      row=row, col=col)

    if loga:
        fig.update_yaxes(type="log")

    fig.update_layout(height=len(countries) * 100, title_text="No. of new cases in each Country")
    fig.update_layout(showlegend=False)
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def epidemicSpan(full_table, radio):
    # first date
    # ==========
    first_date = full_table[full_table['Confirmed'] > 0]
    first_date = first_date.groupby('Country/Region')['Date'].agg(['min']).reset_index()
    # first_date.head()

    # last date
    # =========
    last_date = full_table.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
    last_date = last_date.sum().diff().reset_index()

    mask = last_date['Country/Region'] != last_date['Country/Region'].shift(1)
    last_date.loc[mask, 'Confirmed'] = np.nan
    last_date.loc[mask, 'Deaths'] = np.nan
    last_date.loc[mask, 'Recovered'] = np.nan

    last_date = last_date[last_date['Confirmed'] > 0]
    last_date = last_date.groupby('Country/Region')['Date'].agg(['max']).reset_index()
    # last_date.head()

    # first_last
    # ==========
    first_last = pd.concat([first_date, last_date[['max']]], axis=1)

    # added 1 more day, which will show the next day as the day on which last case appeared
    first_last['max'] = first_last['max'] + timedelta(days=1)

    # no. of days
    first_last['Days'] = first_last['max'] - first_last['min']

    # task column as country
    first_last['Task'] = first_last['Country/Region']

    # rename columns
    first_last.columns = ['Country/Region', 'Start', 'Finish', 'Days', 'Task']

    if radio == "Alphabetically":
        first_last.sort_values('Country/Region', ascending=False, inplace=True)
    elif radio == "Longest Duration":
        # sort by no. of days
        first_last = first_last.sort_values('Days')
    else:
        # sort by no. of days
        first_last = first_last.sort_values('Days')[::-1]

    # first_last.head()

    # visualization
    # =============

    # produce random colors
    clr = ["#" + ''.join([random.choice('0123456789ABC') for j in range(6)]) for i in range(len(first_last))]

    # plot
    fig = ff.create_gantt(first_last, index_col='Country/Region', colors=clr, show_colorbar=False,
                          bar_width=0.2, showgrid_x=True, showgrid_y=True, height=2500)
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def activeCasesTime(full_grouped, radio):
    temp = full_grouped

    if radio == "Alphabetically":
        temp.sort_values('Country/Region', ascending=True, inplace=True)
    elif radio == "Most Cases":
        temp.sort_values('Confirmed', ascending=False, inplace=True)
    else:
        temp.sort_values('Active', ascending=False, inplace=True)

    fig = go.Figure(data=go.Heatmap(
        z=temp['Active'],
        x=temp['Date'],
        y=temp['Country/Region'],
        colorscale=px.colors.sequential.Viridis,
        showlegend=False,
        text=full_grouped['Active']))

    fig.update_layout(yaxis=dict(dtick=1, autorange="reversed"))
    fig.update_layout(height=3000)
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def newCasesTime(full_grouped, radio):
    temp = full_grouped[full_grouped['New cases'] > 0]
    if radio == "Alphabetically":
        temp.sort_values('Country/Region', ascending=True, inplace=True)
    elif radio == "Most Cases":
        temp.sort_values('Confirmed', ascending=False, inplace=True)
    else:
        temp.sort_values('Active', ascending=False, inplace=True)

    fig = px.scatter(temp, x='Date', y='Country/Region', size='New cases', color='New cases', height=3000,
                     color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(yaxis=dict(dtick=1, autorange="reversed"))
    fig.update(layout_coloraxis_showscale=False)
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def caseComposition(full_table):
    full_latest = full_table[full_table['Date'] == max(full_table['Date'])]

    fig1 = px.treemap(full_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True),
                      path=["Country/Region", "Province/State"], values="Confirmed", height=700,
                      title='Number of Confirmed Cases',
                      color_discrete_sequence=px.colors.qualitative.Dark2)
    fig1.data[0].textinfo = 'label+text+value'

    fig2 = px.treemap(full_latest.sort_values(by='Deaths', ascending=False).reset_index(drop=True),
                      path=["Country/Region", "Province/State"], values="Deaths", height=700,
                      title='Number of Deaths reported',
                      color_discrete_sequence=px.colors.qualitative.Dark2)
    fig2.data[0].textinfo = 'label+text+value'
    return [fig1, fig2]


@st.cache(show_spinner=True, allow_output_mutation=True)
def fromCaseN(full_grouped, full_table, slider, loga):
    gt_n = full_grouped[full_grouped['Confirmed'] > slider]['Country/Region'].unique()
    temp = full_table[full_table['Country/Region'].isin(gt_n)]
    temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()
    temp = temp[temp['Confirmed'] > slider]
    # print(temp.head())

    min_date = temp.groupby('Country/Region')['Date'].min().reset_index()
    min_date.columns = ['Country/Region', 'Min Date']
    # print(min_date.head())

    from_100th_case = pd.merge(temp, min_date, on='Country/Region')
    from_100th_case['N days'] = (from_100th_case['Date'] - from_100th_case['Min Date']).dt.days
    # print(from_100th_case.head())

    fig = px.line(from_100th_case, x='N days', y='Confirmed', color='Country/Region',
                  title='Cases from case N° ' + str(slider),
                  height=600)

    if loga:
        fig.update_yaxes(type="log")

    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def percentageCases(full_grouped, day_wise):
    temp = pd.merge(full_grouped[['Date', 'Country/Region', 'Confirmed', 'Deaths']],
                    day_wise[['Date', 'Confirmed', 'Deaths']], on='Date')
    temp['% Confirmed'] = round(temp['Confirmed_x'] / temp['Confirmed_y'], 3) * 100
    temp['% Deaths'] = round(temp['Deaths_x'] / temp['Deaths_y'], 3) * 100
    temp.head()

    fig1 = px.bar(temp, x='Date', y='% Confirmed', color='Country/Region', range_y=(0, 100),
                  title='% of Cases from each country', color_discrete_sequence=px.colors.qualitative.Prism, height=500)

    fig2 = px.bar(temp, x='Date', y='% Deaths', color='Country/Region', range_y=(0, 100),
                  title='% of Deaths from each country', color_discrete_sequence=px.colors.qualitative.Prism,
                  height=500)
    return [fig1, fig2]


@st.cache(show_spinner=True, allow_output_mutation=True)
def dateGraph2(full_grouped):
    fig1 = px.line(full_grouped, x="Date", y="Confirmed", color='Country/Region', height=600,
                   title='Confirmed', color_discrete_sequence=px.colors.cyclical.mygbm)

    # =========================================

    fig2 = px.line(full_grouped, x="Date", y="Deaths", color='Country/Region', height=600,
                   title='Deaths', color_discrete_sequence=px.colors.cyclical.mygbm)

    # =========================================

    fig3 = px.line(full_grouped, x="Date", y="New cases", color='Country/Region', height=600,
                   title='New cases', color_discrete_sequence=px.colors.cyclical.mygbm)

    return [fig1, fig2, fig3]


@st.cache(show_spinner=True, allow_output_mutation=True)
def dateGraph(full_grouped):
    full_grouped.sort_values('Country/Region', inplace=True)
    fig1 = px.bar(full_grouped, x="Date", y="Confirmed", color='Country/Region', height=600,
                  title='Confirmed', color_discrete_sequence=px.colors.cyclical.mygbm)

    # =========================================

    fig2 = px.bar(full_grouped, x="Date", y="Deaths", color='Country/Region', height=600,
                  title='Deaths', color_discrete_sequence=px.colors.cyclical.mygbm)

    # =========================================

    fig3 = px.bar(full_grouped, x="Date", y="New cases", color='Country/Region', height=600,
                  title='New cases', color_discrete_sequence=px.colors.cyclical.mygbm)

    return [fig1, fig2, fig3]


@st.cache(show_spinner=True, allow_output_mutation=True)
def showScatter(country_wise, slider):
    fig = px.scatter(country_wise.sort_values('Deaths', ascending=False).iloc[:slider, :],
                     x='Confirmed', y='Deaths', color='Country/Region', size='Confirmed', height=700,
                     text='Country/Region', log_x=True, log_y=True, title='Deaths vs Confirmed (Scale is in log10)')
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)
    fig.update_layout(xaxis_rangeslider_visible=True)
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def top15(country_wise, selectedCountries):
    selected = (country_wise[country_wise['Country/Region'].isin(selectedCountries)])

    # confirmed - deaths
    temp = (country_wise.sort_values('Confirmed').tail(15).append(selected)).sort_values('Confirmed').drop_duplicates()

    print(temp.head())
    fig_c = px.bar(temp, x="Confirmed", y="Country/Region",
                   text='Confirmed', orientation='h', color_discrete_sequence=[act])
    fig_d = px.bar(
        (country_wise.sort_values('Deaths').tail(15).append(selected)).sort_values('Deaths').drop_duplicates(),
        x="Deaths", y="Country/Region",
        text='Deaths', orientation='h', color_discrete_sequence=[dth])

    # recovered - active
    fig_r = px.bar(
        (country_wise.sort_values('Recovered').tail(15).append(selected)).sort_values('Recovered').drop_duplicates(),
        x="Recovered", y="Country/Region",
        text='Recovered', orientation='h', color_discrete_sequence=[rec])
    fig_a = px.bar(
        (country_wise.sort_values('Active').tail(15).append(selected)).sort_values('Active').drop_duplicates(),
        x="Active", y="Country/Region",
        text='Active', orientation='h', color_discrete_sequence=['#333333'])

    # death - recoverd / 100 cases
    fig_dc = px.bar((country_wise.sort_values('Deaths / 100 Cases').tail(15).append(selected)).sort_values(
        'Deaths / 100 Cases').drop_duplicates(), x="Deaths / 100 Cases", y="Country/Region",
                    text='Deaths / 100 Cases', orientation='h', color_discrete_sequence=['#f38181'])
    fig_rc = px.bar((country_wise.sort_values('Recovered / 100 Cases').tail(15).append(selected)).sort_values(
        'Recovered / 100 Cases').drop_duplicates(), x="Recovered / 100 Cases",
                    y="Country/Region",
                    text='Recovered / 100 Cases', orientation='h', color_discrete_sequence=['#a3de83'])

    # new cases - cases per million people
    fig_nc = px.bar(
        (country_wise.sort_values('New cases').tail(15).append(selected)).sort_values('New cases').drop_duplicates(),
        x="New cases", y="Country/Region",
        text='New cases', orientation='h', color_discrete_sequence=['#c61951'])
    temp = country_wise[country_wise['Population'] > 1000000]
    fig_p = px.bar((temp.sort_values('Cases / Million People').tail(15).append(selected)).sort_values(
        'Cases / Million People').drop_duplicates(), x="Cases / Million People", y="Country/Region",
                   text='Cases / Million People', orientation='h', color_discrete_sequence=['#741938'])

    # week change, percent increase
    fig_wc = px.bar((country_wise.sort_values('1 week change').tail(15).append(selected)).sort_values(
        '1 week change').drop_duplicates(), x="1 week change", y="Country/Region",
                    text='1 week change', orientation='h', color_discrete_sequence=['#004a7c'])
    temp = country_wise[country_wise['Confirmed'] > 100]
    fig_pi = px.bar((temp.sort_values('1 week % increase').tail(15).append(selected)).sort_values(
        '1 week % increase').drop_duplicates(), x="1 week % increase", y="Country/Region",
                    text='1 week % increase', orientation='h', color_discrete_sequence=['#005691'],
                    hover_data=['Confirmed last week', 'Confirmed'])

    # plot
    fig = make_subplots(rows=5, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,
                        subplot_titles=('Confirmed cases', 'Deaths reported', 'Recovered', 'Active cases',
                                        'Deaths / 100 cases', 'Recovered / 100 cases', 'New cases',
                                        'Cases / Million People', '1 week increase', '1 week % increase'))

    fig.add_trace(fig_c['data'][0], row=1, col=1)
    fig.add_trace(fig_d['data'][0], row=1, col=2)
    fig.add_trace(fig_r['data'][0], row=2, col=1)
    fig.add_trace(fig_a['data'][0], row=2, col=2)

    fig.add_trace(fig_dc['data'][0], row=3, col=1)
    fig.add_trace(fig_rc['data'][0], row=3, col=2)
    fig.add_trace(fig_nc['data'][0], row=4, col=1)
    fig.add_trace(fig_p['data'][0], row=4, col=2)

    fig.add_trace(fig_wc['data'][0], row=5, col=1)
    fig.add_trace(fig_pi['data'][0], row=5, col=2)

    fig.update_layout(height=3000)

    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def casesOverTime(day_wise, loga):
    fig_c = px.bar(day_wise, x="Date", y="Confirmed", color_discrete_sequence=[act])
    fig_d = px.bar(day_wise, x="Date", y="Deaths", color_discrete_sequence=[dth])
    fig_r = px.bar(day_wise, x="Date", y="Recovered", color_discrete_sequence=[rec])

    fig = make_subplots(rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.1,
                        subplot_titles=('Confirmed cases', 'Deaths reported',
                                        'Recovered reported'))
    if loga:
        fig.update_yaxes(type="log")
    fig.add_trace(fig_c['data'][0], row=1, col=1)
    fig.add_trace(fig_d['data'][0], row=1, col=2)
    fig.add_trace(fig_r['data'][0], row=1, col=3)

    fig.update_layout(height=480)
    return fig


@st.cache(show_spinner=True, allow_output_mutation=True)
def casesOverTime2(day_wise):
    fig_1 = px.line(day_wise, x="Date", y="Deaths / 100 Cases", color_discrete_sequence=[dth])
    fig_2 = px.line(day_wise, x="Date", y="Recovered / 100 Cases", color_discrete_sequence=[rec])
    fig_3 = px.line(day_wise, x="Date", y="Deaths / 100 Recovered", color_discrete_sequence=['#333333'])

    fig = make_subplots(rows=1, cols=3, shared_xaxes=False,
                        subplot_titles=('Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered'))

    fig.add_trace(fig_1['data'][0], row=1, col=1)
    fig.add_trace(fig_2['data'][0], row=1, col=2)
    fig.add_trace(fig_3['data'][0], row=1, col=3)

    fig.update_layout(height=480)
    return fig


@st.cache(show_spinner=True, allow_output_mutation=True)
def casesOverTime3(day_wise):
    fig_c = px.bar(day_wise, x="Date", y="New cases", color_discrete_sequence=[act])
    fig_n = px.bar(day_wise, x="Date", y="New deaths", color_discrete_sequence=[dth])
    fig_d = px.bar(day_wise, x="Date", y="No. of countries", color_discrete_sequence=['#333333'])

    fig = make_subplots(rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.1,
                        subplot_titles=('New cases everyday',
                                        'New deaths everyday',
                                        'No. of countries'))

    fig.add_trace(fig_c['data'][0], row=1, col=1)
    fig.add_trace(fig_n['data'][0], row=1, col=2)
    fig.add_trace(fig_d['data'][0], row=1, col=3)

    fig.update_layout(height=480)

    return fig


@st.cache(show_spinner=True, allow_output_mutation=True)
def showBarGraph(full_table):
    temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
    temp = temp[temp['Date'] == max(temp['Date'])].reset_index(drop=True)

    tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])
    fig = px.treemap(tm, path=["variable"], values="value", height=225, width=300,
                     color_discrete_sequence=[act, rec, dth])
    fig.data[0].textinfo = 'label+text+value'
    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def showTimeSeries(full_table):
    temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
    temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                     var_name='Case', value_name='Count')

    fig = px.area(temp, x="Date", y="Count", color='Case', height=600,
                  title='Cases over time', color_discrete_sequence=[rec, dth, act])
    fig.update_layout(xaxis_rangeslider_visible=True)

    return [fig]


@st.cache(show_spinner=True, allow_output_mutation=True)
def showWorldMap(country_wise):
    # Confirmed
    fig_c = px.choropleth(country_wise, locations="Country/Region", locationmode='country names',
                          color=np.log(country_wise["Confirmed"]), hover_name="Country/Region",
                          hover_data=['Confirmed'], color_continuous_scale=px.colors.sequential.Purples,
                          title="Confirmed")
    fig_c.update(layout_coloraxis_showscale=False)
    # Deaths
    temp = country_wise[country_wise['Deaths'] > 0]
    fig_d = px.choropleth(temp, locations="Country/Region", locationmode='country names',
                          color=np.log(temp["Deaths"]), hover_name="Country/Region",
                          hover_data=['Deaths'], color_continuous_scale=px.colors.sequential.Reds, title="Deaths")
    fig_d.update(layout_coloraxis_showscale=False)
    return [fig_c, fig_d]


@st.cache(show_spinner=True, allow_output_mutation=True)
def worldTimeGraph(full_grouped):
    # Over the time

    fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names',
                        color=np.log(full_grouped["Confirmed"]),
                        hover_name="Country/Region", animation_frame=full_grouped["Date"].dt.strftime('%Y-%m-%d'),
                        title='Cases over time', color_continuous_scale=px.colors.sequential.Purples)
    fig.update(layout_coloraxis_showscale=False)

    fig2 = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names',
                         color=np.log(full_grouped["Deaths"]),
                         hover_name="Country/Region", animation_frame=full_grouped["Date"].dt.strftime('%Y-%m-%d'),
                         title='Deaths over time', color_continuous_scale=px.colors.sequential.Reds)
    fig2.update(layout_coloraxis_showscale=False)
    return [fig, fig2]


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    '''url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")'''
    with open(path, encoding='utf-8') as file:
        response = file.read()
    return response


@st.cache(show_spinner=True, allow_output_mutation=True)
def loadData():
    full_table = pd.read_csv('input/covid_19_clean_complete.csv', parse_dates=['Date'])

    # ship rows
    ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains(
        'Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table[
                    'Country/Region'].str.contains('MS Zaandam')

    # ship
    ship = full_table[ship_rows]
    # full table
    full_table = full_table[~(ship_rows)]
    # Latest cases from the ships
    ship_latest = ship[ship['Date'] == max(ship['Date'])]

    who_region = {}

    # African Region AFRO
    afro = "Algeria, Angola, Cabo Verde, Eswatini, Sao Tome and Principe, Benin, South Sudan, Western Sahara, Congo (Brazzaville), Congo (Kinshasa), Cote d'Ivoire, Botswana, Burkina Faso, Burundi, Cameroon, Cape Verde, Central African Republic, Chad, Comoros, Ivory Coast, Democratic Republic of the Congo, Equatorial Guinea, Eritrea, Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mauritius, Mozambique, Namibia, Niger, Nigeria, Republic of the Congo, Rwanda, São Tomé and Príncipe, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, Swaziland, Togo, Uganda, Tanzania, Zambia, Zimbabwe"
    afro = [i.strip() for i in afro.split(',')]
    for i in afro:
        who_region[i] = 'afro'

    # Region of the Americas PAHO
    paho = 'Antigua and Barbuda, Argentina, Bahamas, Barbados, Belize, Bolivia, Brazil, Canada, Chile, Colombia, Costa Rica, Cuba, Dominica, Dominican Republic, Ecuador, El Salvador, Grenada, Guatemala, Guyana, Haiti, Honduras, Jamaica, Mexico, Nicaragua, Panama, Paraguay, Peru, Saint Kitts and Nevis, Saint Lucia, Saint Vincent and the Grenadines, Suriname, Trinidad and Tobago, United States, US, Uruguay, Venezuela'
    paho = [i.strip() for i in paho.split(',')]
    for i in paho:
        who_region[i] = 'paho'

    # South-East Asia Region SEARO
    searo = 'Bangladesh, Bhutan, North Korea, India, Indonesia, Maldives, Myanmar, Burma, Nepal, Sri Lanka, Thailand, Timor-Leste'
    searo = [i.strip() for i in searo.split(',')]
    for i in searo:
        who_region[i] = 'searo'

    # European Region EURO
    euro = 'Albania, Andorra, Greenland, Kosovo, Holy See, Liechtenstein, Armenia, Czechia, Austria, Azerbaijan, Belarus, Belgium, Bosnia and Herzegovina, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Georgia, Germany, Greece, Hungary, Iceland, Ireland, Israel, Italy, Kazakhstan, Kyrgyzstan, Latvia, Lithuania, Luxembourg, Malta, Monaco, Montenegro, Netherlands, North Macedonia, Norway, Poland, Portugal, Moldova, Romania, Russia, San Marino, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Tajikistan, Turkey, Turkmenistan, Ukraine, United Kingdom, Uzbekistan'
    euro = [i.strip() for i in euro.split(',')]
    for i in euro:
        who_region[i] = 'euro'

    # Eastern Mediterranean Region EMRO
    emro = 'Afghanistan, Bahrain, Djibouti, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Pakistan, Palestine, West Bank and Gaza, Qatar, Saudi Arabia, Somalia, Sudan, Syria, Tunisia, United Arab Emirates, Yemen'
    emro = [i.strip() for i in emro.split(',')]
    for i in emro:
        who_region[i] = 'emro'

    # Western Pacific Region WPRO
    wpro = 'Australia, Brunei, Cambodia, China, Cook Islands, Fiji, Japan, Kiribati, Laos, Malaysia, Marshall Islands, Micronesia, Mongolia, Nauru, New Zealand, Niue, Palau, Papua New Guinea, Philippines, South Korea, Samoa, Singapore, Solomon Islands, Taiwan, Taiwan*, Tonga, Tuvalu, Vanuatu, Vietnam'
    wpro = [i.strip() for i in wpro.split(',')]
    for i in wpro:
        who_region[i] = 'wpro'

    full_table['WHO Region'] = full_table['Country/Region'].map(who_region)
    full_table[full_table['WHO Region'].isna()]['Country/Region'].unique()

    # fixing Country values
    full_table.loc[full_table['Province/State'] == 'Greenland', 'Country/Region'] = 'Greenland'

    # Active Case = confirmed - deaths - recovered
    full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

    # replacing Mainland china with just China
    full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')

    # filling missing values
    full_table[['Province/State']] = full_table[['Province/State']].fillna('')
    full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[
        ['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)

    # fixing datatypes
    full_table['Recovered'] = full_table['Recovered'].astype(int)

    # Grouped by day, country
    # =======================

    full_grouped = full_table.groupby(['Date', 'Country/Region'])[
        'Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

    # new cases ======================================================
    temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
    temp = temp.sum().diff().reset_index()

    mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

    temp.loc[mask, 'Confirmed'] = np.nan
    temp.loc[mask, 'Deaths'] = np.nan
    temp.loc[mask, 'Recovered'] = np.nan

    # renaming columns
    temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']
    # =================================================================

    # merging new values
    full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])

    # filling na with 0
    full_grouped = full_grouped.fillna(0)

    # fixing data types
    cols = ['New cases', 'New deaths', 'New recovered']
    full_grouped[cols] = full_grouped[cols].astype('int')

    full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x < 0 else x)
    full_grouped['WHO Region'] = full_grouped['Country/Region'].map(who_region)

    # Day wise
    # ========

    # table
    day_wise = full_grouped.groupby('Date')[
        'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths'].sum().reset_index()

    # number cases per 100 cases
    day_wise['Deaths / 100 Cases'] = round((day_wise['Deaths'] / day_wise['Confirmed']) * 100, 2)
    day_wise['Recovered / 100 Cases'] = round((day_wise['Recovered'] / day_wise['Confirmed']) * 100, 2)
    day_wise['Deaths / 100 Recovered'] = round((day_wise['Deaths'] / day_wise['Recovered']) * 100, 2)

    # no. of countries
    day_wise['No. of countries'] = full_grouped[full_grouped['Confirmed'] != 0].groupby('Date')[
        'Country/Region'].unique().apply(len).values

    # fillna by 0
    cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']
    day_wise[cols] = day_wise[cols].fillna(0)

    # Country wise
    # ============

    # getting latest values
    country_wise = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])].reset_index(drop=True).drop('Date',
                                                                                                               axis=1)

    # group by country
    country_wise = country_wise.groupby('Country/Region')[
        'Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()

    # per 100 cases
    country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths'] / country_wise['Confirmed']) * 100, 2)
    country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered'] / country_wise['Confirmed']) * 100, 2)
    country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths'] / country_wise['Recovered']) * 100, 2)

    cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']
    country_wise[cols] = country_wise[cols].fillna(0)
    country_wise['WHO Region'] = country_wise['Country/Region'].map(who_region)

    # load population dataset
    pop = pd.read_csv("input/population_by_country_2020.csv")

    # select only population
    pop = pop.iloc[:, :2]

    # rename column names
    pop.columns = ['Country/Region', 'Population']

    # merged data
    country_wise = pd.merge(country_wise, pop, on='Country/Region', how='left')

    # update population
    cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia',
            'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines',
            'Taiwan*', 'US', 'West Bank and Gaza', 'Sao Tome and Principe']
    pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000,
            53109, 110854, 23806638, 330541757, 4543126, 219159]
    for c, p in zip(cols, pops):
        country_wise.loc[country_wise['Country/Region'] == c, 'Population'] = p

    # missing values
    # country_wise.isna().sum()
    # country_wise[country_wise['Population'].isna()]['Country/Region'].tolist()

    # Cases per population
    country_wise['Cases / Million People'] = round((country_wise['Confirmed'] / country_wise['Population']) * 1000000)

    #Deaths per population
    country_wise['Deaths / Million People'] = round((country_wise['Deaths'] / country_wise['Population']) * 1000000)

    today = full_grouped[full_grouped['Date'] == max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)[
        ['Country/Region', 'Confirmed']]
    last_week = \
        full_grouped[full_grouped['Date'] == max(full_grouped['Date']) - timedelta(days=7)].reset_index(drop=True).drop(
            'Date', axis=1)[['Country/Region', 'Confirmed']]

    temp = pd.merge(today, last_week, on='Country/Region', suffixes=(' today', ' last week'))

    # temp = temp[['Country/Region', 'Confirmed last week']]
    temp['1 week change'] = temp['Confirmed today'] - temp['Confirmed last week']

    temp = temp[['Country/Region', 'Confirmed last week', '1 week change']]

    country_wise = pd.merge(country_wise, temp, on='Country/Region')

    country_wise['1 week % increase'] = round(country_wise['1 week change'] / country_wise['Confirmed last week'] * 100,
                                              2)

    return (full_table, full_grouped, day_wise, country_wise)


def getDataToCsv():

    conf_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recv_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    # extract dates
    dates = conf_df.columns[4:]

    # melt dataframes in longer format
    conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                                value_vars=dates, var_name='Date', value_name='Confirmed')

    deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                                    value_vars=dates, var_name='Date', value_name='Deaths')

    recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                                value_vars=dates, var_name='Date', value_name='Recovered')

    recv_df_long = recv_df_long[recv_df_long['Country/Region'] != 'Canada']

    full_table = pd.merge(left=conf_df_long, right=deaths_df_long, how='left',
                          on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])
    full_table = pd.merge(left=full_table, right=recv_df_long, how='left',
                          on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long'])

    # renaming countries, regions, provinces
    full_table['Country/Region'] = full_table['Country/Region'].replace('Korea, South', 'South Korea')

    # removing canada's recovered values
    full_table = full_table[full_table['Province/State'].str.contains('Recovered') != True]

    # removing county wise data to avoid double counting
    full_table = full_table[full_table['Province/State'].str.contains(',') != True]

    # new values
    feb_12_conf = {'Hubei': 34874}

    # changing values
    full_table = change_val('2/12/20', 'Province/State', 'Confirmed', feb_12_conf, full_table)

    who_region = {}

    # African Region AFRO
    afro = "Algeria, Angola, Cabo Verde, Eswatini, Sao Tome and Principe, Benin, South Sudan, Western Sahara, Congo (Brazzaville), Congo (Kinshasa), Cote d'Ivoire, Botswana, Burkina Faso, Burundi, Cameroon, Cape Verde, Central African Republic, Chad, Comoros, Ivory Coast, Democratic Republic of the Congo, Equatorial Guinea, Eritrea, Ethiopia, Gabon, Gambia, Ghana, Guinea, Guinea-Bissau, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mauritius, Mozambique, Namibia, Niger, Nigeria, Republic of the Congo, Rwanda, São Tomé and Príncipe, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, Swaziland, Togo, Uganda, Tanzania, Zambia, Zimbabwe"
    afro = [i.strip() for i in afro.split(',')]
    for i in afro:
        who_region[i] = 'afro'

    # Region of the Americas PAHO
    paho = 'Antigua and Barbuda, Argentina, Bahamas, Barbados, Belize, Bolivia, Brazil, Canada, Chile, Colombia, Costa Rica, Cuba, Dominica, Dominican Republic, Ecuador, El Salvador, Grenada, Guatemala, Guyana, Haiti, Honduras, Jamaica, Mexico, Nicaragua, Panama, Paraguay, Peru, Saint Kitts and Nevis, Saint Lucia, Saint Vincent and the Grenadines, Suriname, Trinidad and Tobago, United States, US, Uruguay, Venezuela'
    paho = [i.strip() for i in paho.split(',')]
    for i in paho:
        who_region[i] = 'paho'

    # South-East Asia Region SEARO
    searo = 'Bangladesh, Bhutan, North Korea, India, Indonesia, Maldives, Myanmar, Burma, Nepal, Sri Lanka, Thailand, Timor-Leste'
    searo = [i.strip() for i in searo.split(',')]
    for i in searo:
        who_region[i] = 'searo'

    # European Region EURO
    euro = 'Albania, Andorra, Greenland, Kosovo, Holy See, Liechtenstein, Armenia, Czechia, Austria, Azerbaijan, Belarus, Belgium, Bosnia and Herzegovina, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Georgia, Germany, Greece, Hungary, Iceland, Ireland, Israel, Italy, Kazakhstan, Kyrgyzstan, Latvia, Lithuania, Luxembourg, Malta, Monaco, Montenegro, Netherlands, North Macedonia, Norway, Poland, Portugal, Moldova, Romania, Russia, San Marino, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Tajikistan, Turkey, Turkmenistan, Ukraine, United Kingdom, Uzbekistan'
    euro = [i.strip() for i in euro.split(',')]
    for i in euro:
        who_region[i] = 'euro'

    # Eastern Mediterranean Region EMRO
    emro = 'Afghanistan, Bahrain, Djibouti, Egypt, Iran, Iraq, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Pakistan, Palestine, West Bank and Gaza, Qatar, Saudi Arabia, Somalia, Sudan, Syria, Tunisia, United Arab Emirates, Yemen'
    emro = [i.strip() for i in emro.split(',')]
    for i in emro:
        who_region[i] = 'emro'

    # Western Pacific Region WPRO
    wpro = 'Australia, Brunei, Cambodia, China, Cook Islands, Fiji, Japan, Kiribati, Laos, Malaysia, Marshall Islands, Micronesia, Mongolia, Nauru, New Zealand, Niue, Palau, Papua New Guinea, Philippines, South Korea, Samoa, Singapore, Solomon Islands, Taiwan, Taiwan*, Tonga, Tuvalu, Vanuatu, Vietnam'
    wpro = [i.strip() for i in wpro.split(',')]
    for i in wpro:
        who_region[i] = 'wpro'

    full_table['WHO Region'] = full_table['Country/Region'].map(who_region)
    full_table[full_table['WHO Region'].isna()]['Country/Region'].unique()

    full_table.to_csv('input/covid_19_clean_complete.csv', index=False)


# function to change value
def change_val(date, ref_col, val_col, dtnry, full_table):
    for key, val in dtnry.items():
        full_table.loc[(full_table['Date']==date) & (full_table[ref_col]==key), val_col] = val
    return full_table

def checkLast(full_table):
    last = full_table.iloc[-1]['Date']
    today = date.today()
    diff = today - last.date()
    if diff.days == 0:
        return "Today"
    elif diff.days == 1:
        return "Yesterday"
    return str(diff.days) + " days ago"


def get_session_id():
    # Hack to get the session object from Streamlit.

    ctx = ReportThread.get_report_ctx()

    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
                # Streamlit < 0.54.0
                (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
                or
                # Streamlit >= 0.54.0
                (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object"
            'Are you doing something fancy with threads?')

    return id(this_session)

if __name__ == "__main__":
    main()
