import streamlit as st
import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
import pandas as pd
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm
import plotly.express as px
from datetime import datetime
import functools
from wordcloud import WordCloud
from annotated_text import annotated_text


# 문장 간의 유사성 분석을 위한 모듈
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode

import warnings
warnings.simplefilter("ignore")


# 페이지 기본 설정
st.set_page_config(
    page_title='NewScan',
    page_icon=':newspaper:',
    layout='wide'
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
chart = functools.partial(st.plotly_chart, use_container_width=True)
COMMON_ARGS = {
    "color": "symbol",
    "color_discrete_sequence": px.colors.sequential.Greens,
    "hover_data": [
        "account_name",
        "percent_of_account",
        "quantity",
        "total_gain_loss_dollar",
        "total_gain_loss_percent",
    ],
}

####
# 문장 유사도 산출을 위한 cosine 유사도 계산 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 제목 코사인 유사도를 이용한 유사 기사 추천
def get_recommendations(heads, cosine_sim, count=10):
    # 선택한 기사의 타이틀로부터 해당 기사의 인덱스를 받아온다.
    idx = title_to_index[heads]

    # 해당 기사와 모든 기사의 유사도를 가져온다.
    sim_scores = [i for i in enumerate(cosine_sim[idx])]

    # 유사도에 따라 기사들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 유사도가 0.3 이상인 기사들만 가져온다.
    sim_scores = [x for x in sim_scores if x[1] >= 0.3]

    # 가장 유사한 5개의 기사를 받아온다.
    sim_scores = sim_scores[:count]

    # 가장 유사한 5개의 기사의 인덱스를 얻는다.
    news_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한  5개의 기사의 제목을 리턴한다.
    return [df['제목'].iloc[news_indices], sim_scores]


# 데이터셋 준비
@st.cache_data
def load_data():
    data = pd.read_csv('data/news_total.csv', index_col=0)
    return data
df = load_data()



####

# 타이틀
st.title(":pager: NEWSANS")

# 뉴스 일자 필터링
st.subheader("뉴스 일자")
start_time = st.date_input(
    'Select date',
    value=(datetime(2023, 5, 21), datetime(2023, 5, 28))
)

# 일자 필터링
df = df[(df['일자'] >= str(start_time[0])) & (df['일자'] <= str(start_time[1]))]

#####
df2 = df[['인물', '위치', '기관']].fillna('')
df3 = df2.인물 + df2.위치 + df2.기관
# 벡터화
tfidf = TfidfVectorizer()
abc_tfidf_matrix = tfidf.fit_transform(df3)
k_men = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1], reverse=True)
k_men = list(map(lambda x: x[0], k_men))


day_sel_i = str(start_time[0])
daily_data = df[df['일자'] == day_sel_i].reset_index()

y = start_time[0].year
m = start_time[0].month
d0 = start_time[0].day
d1 = start_time[1].day

st.write('---')

####

# 옵션 선택 1,2
col01, col02, = st.columns(2)
# 키워드 검색
with col01:
    st.subheader("키워드 검색")
    keyword_selections = st.multiselect(
    "Select 키워드 to View", options = k_men, default = ['현대차'],
    label_visibility='collapsed',
    max_selections=1)

# 언론사 필터링
with col02:
    st.subheader("언론사")
    accounts = list(df['언론사'].unique())
    account_selections = st.multiselect(
    "Select 언론사 to View", options = accounts, default = ['조선일보','중앙일보','한겨레','한국경제','동아일보', '머니투데이'] ,
    label_visibility='collapsed')

st.write("---")

col11, col12, col13 = st.columns(3)

# 감성 필터링
with col11:
    st.markdown(":moyai:**감성 필터링**")
    filling_selections = st.select_slider(
        "감정 분류",
        options=['긍정', '중립', '부정'],
        value=('긍정', '중립'),
        label_visibility='collapsed',
    )

# 뉴스 제목 - 본문 일치도 필터링
with col12:
    st.markdown(":question:**제목-본문 일치도**")
    cosine_sim = st.slider(
        "0 : 불일치 -  1 : 일치",
        min_value= 0.0,
        max_value= 1.0,
        value=(0.25, 0.75),
        step=0.01,
        label_visibility='collapsed',
    )

# 낚시기사 필터링
with col13:
    st.markdown(":fishing_pole_and_fish: **낚시성 기사 필터링**")
    fishing = st.slider(
        "0 : 의심  -  1 : 신뢰",
        min_value= 0.0,
        max_value= 1.0,
        value=(0.25, 0.75),
        step=0.01,
        label_visibility='collapsed'
    )

st.write("---")
# 언론사 필터링
df = df[df['언론사'].isin(account_selections)]
# 키워드 필터링
df = df[df['키워드'].str.contains(keyword_selections[0])]
# 감성 필터링
df = df[df['Senti_label'].isin(filling_selections)]
# 뉴스 제목-본문 일치도 필터링
df = df[(df['Sim_score'] >= cosine_sim[0]) & (df['Sim_score'] <= cosine_sim[1])]
# 낚시기사 필터링
df = df[(df['fishing_score'] >= fishing[0]) & (df['fishing_score'] <= fishing[1])]
df.sort_values(by='Sim_score', inplace=True, ascending=False)

####
def create_link(title, url: str) -> str:
    return f'''<a href="{url}">{title}</a>'''

# 링크생성
def summary_link(url: str) -> str:
    return f'''<a href="{url}">📰</a>'''


# 긍정 라벨 이모지
def senti_label(Senti_label) :
    if Senti_label == '긍정':
        emoji = '🔵'
    elif Senti_label == '부정':
        emoji = '🔴'
    else:
        emoji = '🟡'
    return emoji + Senti_label

# 피싱 라벨 이모지
def fishing_label(fishing) :
    if fishing == '양호':
        emoji2 = '양호'
    else:
        emoji2 = '🐟피싱의심'
    return emoji2


df['기사'] = [create_link(title, url) for title, url in zip(df['제목'], df['URL'])]
news = df[['기사']].copy()

df['Link'] = [summary_link(url) for url in df["URL"]]
df['긍부정 여부'] = df.Senti_label.map(lambda x: senti_label(x))
df['피싱기사 여부'] = df['낚시기사'].map(lambda x: fishing_label(x))
df['제목-본문 일치도(%)'] = df.Sim_score.map(lambda x: str(np.round(x*100,1))+'%')


### 본문

# 최종 출력 데이터프레임
n = 10

### 구분

st.subheader(f':date:{y}년 {m}월 {d0}일 ~{m}월 {d1}일 ')
st.markdown(f'**[ {keyword_selections[0]} ]** 관련 경제 기사 : **총 {len(df)}건**')


###
col1, col2 = st.columns([0.4,0.6])

# 긍부정 비율 pychart
with col1:
    st.markdown('')
    st.markdown(f'**긍부정 비율**')

    wedgeprops = {'width': 0.3, 'edgecolor': 'w', 'linewidth': 5}

    temp_df = pd.DataFrame(df['Senti_label'].value_counts())

    # colors = ['#ffc000', '#3e8df4', '#ff9999']
    color_map = {'중립': '#ffc000', '긍정': '#3e8df4', '부정': '#ff9999'}
    temp_df["color"] = temp_df.index.map(color_map)
    colors = temp_df['color'].values
    wdges, labels, autopct = plt.pie(temp_df['Senti_label'], labels=temp_df.index, startangle=90,
                                     radius=1, autopct='%.1f%%', colors=colors, wedgeprops=wedgeprops,
                                     textprops={'font': fontprop}, pctdistance=1.2, labeldistance=1.4)

    plt.setp([labels, autopct], fontsize=15)
    plt.tight_layout()
    st.pyplot(plt)

# 워드 클라우드
with col2:
    st.markdown('')
    st.markdown(f'**관련 키워드**')
    st.caption('- 언급량은 키워드의 크기에 비례함')

    col_l, col_c, col_r = st.columns([0.05, 0.9, 0.05])

    with col_c:
        # 인물, 위치, 기관 컬럼 내 결측치 변환
        list = ['인물', '위치', '기관']
        for l in list:
            df[l].replace({np.nan: ''}, inplace=True)

        # 키워드 컬럼 내 중복 키워드 제거
        for i in range(len(daily_data)):
            temp = ''
            for word in daily_data.iloc[i]['키워드'].split(','):
                if word not in temp:
                    temp += word + ','
            df.at[i, '키워드'] = temp


        def word_count_dic(daily_data, col):
            cnt = {}
            for i in range(len(daily_data)):
                for word in daily_data.iloc[i][col].split(','):
                    if word in cnt:
                        cnt[word] += 1
                    else:
                        cnt[word] = 1
            del cnt['']
            return cnt
        # total_key = word_count_dic(daily_data, '키워드')
        # total_key = dict(sorted(total_key.items(), reverse=True, key=lambda x: x[1]))
        # people = word_count_dic(daily_data, '인물')
        # location = word_count_dic(daily_data, '위치')
        # organization = word_count_dic(daily_data, '기관')

        people = df.인물.dropna()
        location = df.위치.dropna()
        organization = df.기관.dropna()
        etc = df.키워드.dropna()

        # 벡터화
        tfidf = TfidfVectorizer(min_df=5, max_features=20)


        def tfidf_fit_trans(df):
            ab_tfidf_matrix = tfidf.fit_transform(df)
            key_dict = dict(sorted(tfidf.vocabulary_.items(), key=lambda x: x[1], reverse=True))
            return key_dict


        pp_keyword = tfidf_fit_trans(people)
        loc_keyword = tfidf_fit_trans(location)
        org_keyword = tfidf_fit_trans(organization)

        keyword_total = pp_keyword | loc_keyword | org_keyword
        keyword_total = dict(sorted(keyword_total.items(), reverse=True, key=lambda x: x[1]))

        # font_path = r"C:\Windows\Fonts\malgun.ttf"
        wordcloud = WordCloud(font_path='font/NanumGothic.ttf',
                              background_color='white',
                              margin=15,
                              colormap='PuBu',
                              prefer_horizontal=1,
                              min_font_size=10,
                              max_font_size=25,
                              max_words=70,
                              contour_color=None).fit_words(keyword_total)


        class SimpleGroupedColorFunc(object):
            def __init__(self, color_to_words, default_color):
                self.word_to_color = {word: color
                                      for (color, words) in color_to_words.items()
                                      for word in words}

                self.default_color = default_color

            def __call__(self, word, **kwargs):
                return self.word_to_color.get(word, self.default_color)


        color_to_words = {
            '#8fd9b6': [i for i in keyword_total.keys() if i in pp_keyword.keys()],
            '#d395d0': [i for i in keyword_total.keys() if i in loc_keyword.keys()],
            '#ff9999': [i for i in keyword_total.keys() if i in org_keyword.keys()]
        }

        default_color = '#8fd9b6'

        grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

        # 키워드별 색상 변경
        wordcloud.recolor(color_func=grouped_color_func)

        # 그래프
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        st.pyplot(plt)

        annotated_text('범례 : ',
            ("인물", '', "#8fd9b6"),
                       ' ',
            ("장소", '', "#d395d0"),
                       ' ',
            ("기관", '', "#ff9999"),
            # ("기타 키워드 ", '', "#8fd9b6"),
        )


st.markdown('---')


##


## 뉴스기사
st.subheader(f':newspaper: NEWSCAN')
st.markdown(":point_down: 해당 기사를 클릭하면 분석내용을 확인하실 수 있습니다")

# select the columns you want the users to see
gb = GridOptionsBuilder.from_dataframe(df[['일자', '제목', '긍부정 여부', '피싱기사 여부', '제목-본문 일치도(%)',]])
# configure selection
gb.configure_selection(selection_mode="single", use_checkbox=True)
gb.configure_side_bar()
gridOptions = gb.build()

data = AgGrid(df.head(n),
              gridOptions=gridOptions,
              enable_enterprise_modules=True,
              allow_unsafe_jscode=True,
              update_mode=GridUpdateMode.SELECTION_CHANGED,
              columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)

st.caption('-긍부정 여부 : 분류 결과 가장 높은 확률(probablity)을 가진 라벨값(중립/긍정/부정)을 제시.  \n '
           '-제목 본문 일치도 : 제목과 본문의 일치 정도를 퍼센트로 제시.  \n'
           '-피싱기사 여부 : 피싱 확률이 0.5 이상인 경우, 🐟피싱 의심.')
selected_rows = data["selected_rows"]

####
st.write("---")

####

with st.spinner('Wait for it...'):
    time.sleep(3)

###############
# 분석기사 선택시
if len(selected_rows) != 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="감성 필터", value=f"{selected_rows[0]['Senti_label']}", delta=f"{selected_rows[0]['proba']:.3f}")

    with col2:
        st.metric(label="본문 일치도", value=f"{selected_rows[0]['본문일치도']}", delta=f"{selected_rows[0]['Sim_score']:.3f}")

    with col3:
        st.metric(label="낚시기사 필터", value=f"{selected_rows[0]['낚시기사']}", delta=f"{selected_rows[0]['fishing_score']:.3f}")

    st.write('---')
    st.markdown(":clipboard: **기사요약**")
    st.text_area(f'**기사 제목 : {selected_rows[0]["제목"]}**',
                 value = f"{str(selected_rows[0]['summary'])}",
                 label_visibility='visible',
                 height=300
                 )
    st.caption('- Gensim 모델을 활용하여 본문을 300자 이내로 요약 \n '
               )



    st.write('---')

    ### 유사도 추출
    del tfidf, wordcloud,
    def sim(df):
        df = df.reset_index(drop=True)
        title_to_index = dict(zip(df['제목'], df.index))
        ab = ["".join(h).replace(',', ' ') for h in df.키워드]

        # 벡터화
        tfidf = TfidfVectorizer()
        abc_tfidf_matrix = tfidf.fit_transform(ab)

        # 기사제목, 본문별 Cosine 유사도 산출
        abc_cosine_sim = cosine_similarity(abc_tfidf_matrix, abc_tfidf_matrix)

        return title_to_index, abc_cosine_sim


    title_to_index, sim = sim(df)
    st.markdown(":printer: **유사 기사 Recommand Top 10**")

    recom = get_recommendations(selected_rows[0]['제목'], sim)
    sim_df = pd.DataFrame(recom[0])
    sim_df = sim_df.merge(df, on='제목')[['제목','일자','언론사','기고자', '피싱기사 여부','URL']]
    # sim_df['유사도'] = np.array(recom[1])[:,1]
    sim_df.reset_index(drop=True, inplace=True)
    st.dataframe(sim_df[1:],
                 use_container_width=True,
                 column_config={
                     "URL": st.column_config.LinkColumn("원문 링크")
                 }
    )
    st.caption('- 선택 기사와 유사도가 높은 순서로 정렬 \n '
           '- 유사도 산출방식은 Cosin_similarity 를 사용  \n'
               )


