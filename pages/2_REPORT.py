import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from numpy import dot
from numpy.linalg import norm
import time
import platform
import matplotlib.font_manager as fm
from datetime import datetime
import functools


# 페이지 기본 설정
st.set_page_config(
    page_title = 'trend',
    page_icon = ':newspaper:',
    layout= 'wide',
)


# 한글 폰트 지정
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':  # 맥OS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # 윈도우
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system...  sorry~~~')

# 한글폰트지정 2
fontprop = fm.FontProperties(fname="font/NanumGothic.ttf")



# 전처리 된 데이터 불러오기
df = pd.read_csv('data/news_total.csv')


# 날짜 설정
select_date = st.date_input('날짜를 선택해주세요.',
                            value=(datetime(2023, 5, 21), datetime(2023, 5, 28))
                            )
st.caption('현재는 2023년 5월 경제 기사에 한하여 제한적으로 서비스되고 있습니다.')

start_date = select_date[0] # 시작 날짜
end_date = select_date[1]   # 종료 날짜


# 날짜 필터링
filter_date = (df['일자'] >= str(start_date)) & (df['일자'] <= str(end_date))
df_date = df.loc[filter_date]


st.markdown('-----------------------')



# 주요 키워드

st.header('🔆**주요 키워드**')
st.caption('선택된 기간의 경제 기사에서 많이 언급된 상위 20개의 키워드를 제공합니다')

# 날짜 필터링 된 데이터 파일
df_date = df[filter_date].reset_index()

# 인물, 위치, 기관 컬럼 내 결측치 변환
list = ['인물', '위치', '기관']
for l in list:
    df_date[l].replace({np.nan: ''}, inplace=True)

# 키워드 컬럼 내 중복 키워드 제거
for i in range(len(df_date)):
    temp = ''
    for word in df_date.iloc[i]['키워드'].split(','):
        if word not in temp:
            temp += word + ','
    df_date.at[i, '키워드'] = temp

# 키워드 추출 함수
def word_count_dic(df_date, col):
    cnt = {}
    for i in range(len(df_date)):
        for word in df_date.iloc[i][col].split(','):
            if word in cnt:
                cnt[word] += 1
            else:
                cnt[word] = 1
    del cnt['']
    return cnt

# 주요 키워드 추출
people = word_count_dic(df_date, '인물')
location = word_count_dic(df_date, '위치')
organization = word_count_dic(df_date,'기관')
total_key = people | location | organization
total_key_df = pd.DataFrame(total_key, index=['키워드 수']).T
# 주요 키워드 상위 20개 추출
total_key_df = total_key_df.sort_values(by='키워드 수', ascending=False).head(20)
total_key_list = total_key_df.sort_values(by='키워드 수', ascending=False).head(50)
st.bar_chart(total_key_df)

# 주요 키워드 중 가장 언급량이 많았던 키워드 추출
most_keyword = total_key_df.sort_values(by='키워드 수',ascending=False).head(1).index.to_list()
st.info(f'##### 가장 언급량이 많았던 키워드: {most_keyword[0]}')
st.caption('키워드가 포함된 경제 뉴스 기사의 긍부정 비율과 기사 건 수 변화량을 제공합니다')

st.subheader("키워드 검색")
sel_keyword = st.multiselect("Select 키워드 to View", options=total_key_list.index, default='삼성전자',
                                label_visibility='collapsed', max_selections=1)

# 키워드의 긍부정 비율 확인을 위한 데이터
senti_df = df_date[['일자', '키워드', 'Senti_label']]

senti_df = senti_df[senti_df['키워드'].str.contains(sel_keyword[0], na = False)]
senti_group = senti_df.groupby(by='Senti_label').count()
senti_group.reset_index(drop=False, inplace=True)


col01, col02, = st.columns([0.4, 0.6])
with col01: # 주요 키워드의 긍부정 비율의 pie chart
    st.markdown(f'- 키워드 \"{sel_keyword[0]}\"의 긍부정 비율')

    fig = plt.figure()
    ax = fig.add_subplot()
    colors = ['#3e8df4','#ff9999', '#ffc000']
    ax.pie(senti_group['키워드'],
           labels=senti_group['Senti_label'],
           autopct='%.2f%%',
           colors = colors)
    plt.show()
    st.pyplot(plt)

with col02: # 주요 키워드의 관련 건 수 변화량 관찰을 위한 line chart
    st.markdown(f'- 키워드 \"{sel_keyword[0]}\"의 관련 기사 건 수 변화량')
    senti_date = senti_df.groupby('일자')['키워드'].count()
    senti_date = senti_date.to_frame()
    st.line_chart(senti_date)

st.write('')

st.markdown(f'- 키워드 \"{sel_keyword[0]}\"의 긍정, 중립, 부정 기사 건수 변화량')
pos_date = senti_df[senti_df['Senti_label'] == '긍정'].groupby('일자')['키워드'].count().T
pos_date.rename(' 긍정', inplace=True)
neu_date = senti_df[senti_df['Senti_label'] == '중립'].groupby('일자')['키워드'].count().T
neu_date.rename(' 중립', inplace=True)
neg_date = senti_df[senti_df['Senti_label'] == '부정'].groupby('일자')['키워드'].count().T
neg_date.rename('부정', inplace=True)
total_data = pd.concat([pos_date, neg_date, neu_date], axis=1)

st.line_chart(total_data)

st.write('')
st.write('')
st.write('')
st.markdown('-----')
# 주제 분류별 기사량
st.markdown('#### 📚 주제 분류별 키워드')
st.caption('가장 언급량이 많았던 키워드가 포함된 경제 뉴스 기사의 주제 분류별 기사량을 제공합니다.')

count_list = []
keywords = ['경제일반', '국제경제', '금융_재테크', '무역', '반도체', '부동산', '산업_기업', '서비스_쇼핑', '외환', '유통', '자동차', '자원', '증권_증시', '취업_창업']

cat_df = df_date[['일자', '키워드', '통합분류']]
cat_df['통합분류'] = cat_df['통합분류'].str.strip()
cat_df['주제 분류'] = cat_df['통합분류'].str.split(' ').str[0]

most_keyword = total_key_df.sort_values(by='키워드 수',ascending=False).head(1).index.to_list()
st.markdown(f'- 가장 언급량이 많았던 키워드\"{most_keyword[0]}\"의 주제 분류별 기사량')

cat_df = cat_df[cat_df['키워드'].str.contains(most_keyword[0], na = False)]
cat_group = cat_df.groupby(by='주제 분류').count()
cat_group.reset_index(drop=False, inplace=True)

cat_group.set_index(cat_group['주제 분류'], inplace=True)
cat_group = cat_group['키워드']

st.bar_chart(cat_group)

st.write('')
st.write('--------------------------')
st.write('')



# 테마별 키워드

st.subheader('🔍테마별 키워드')
st.caption('선택된 기간 중 많이 언급된 상위 10개의 테마별 키워드와 각 테마의 언급량이 가장 많은 키워드 및 관련 정보를 제공합니다 ')

# 테마별 키워드 추출 데이터
people = word_count_dic(df_date, '인물')
location = word_count_dic(df_date, '위치')
organization = word_count_dic(df_date, '기관')

# 테마별 키워드의 긍부정 비율과 관련 기사 건 수 확인을 위한 함수
def get_keyword(data, theme):
    data = pd.DataFrame(data, index=['키워드 수']).T
    srt_data = data.sort_values(by='키워드 수', ascending=False).head(10)
    st.bar_chart(srt_data)

    most_keyword = srt_data.sort_values(by='키워드 수', ascending=False).head(1).index.to_list()
    st.success(f'언급량이 많았던 **{theme}** 키워드: {most_keyword[0]}')

    col01, col02, = st.columns([0.4, 0.6])
    with col01:
        st.markdown(f'- 키워드 \"{most_keyword[0]}\"의 긍부정 비율')

        senti_df = df_date[['일자', '키워드', 'Senti_label']]
        senti_df = senti_df[senti_df['키워드'].str.contains(most_keyword[0], na=False)]
        senti_group = senti_df.groupby(by='Senti_label').count()
        senti_group.reset_index(drop=False, inplace=True)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot()

        ax.pie(senti_group['키워드'],
               labels=senti_group['Senti_label'],
               autopct='%.2f%%',
               colors = colors)
        plt.show()
        st.pyplot(plt)

    with col02:
        st.markdown(f'- 키워드 \"{most_keyword[0]}\"의 관련 기사 건 수 변화량')
        senti_date = senti_df.groupby('일자')['키워드'].count()
        senti_date = senti_date.to_frame()
        st.line_chart(senti_date)


# 테마별 키워드 및 관련 정보
tab1, tab2, tab3 = st.tabs(['장소', '인물', '기관'])

with tab1:
    get_keyword(location, '장소')

with tab2:
    get_keyword(people, '인물')

with tab3:
    get_keyword(organization, '기관')

