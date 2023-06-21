# -*- coding: utf-8 -*-
# 모듈 설치 : pip install wordcloud, pip install plotly, pip install st-annotated-text

import streamlit as st
from PIL import Image
import warnings

warnings.simplefilter("ignore")

# 페이지 기본 설정
st.set_page_config(
    page_title='Service Info',
    page_icon='	:card_index:',
    layout='wide'
)

# 서비스 소개
# col_logo, col_info = st.columns([0.2,0.8])
# with col_logo:
#     st.image(Image.open('data/Logo.png'))
# with col_info:
#     st.markdown('  \n '
#                 '  \n ')
#     st.title('NewScan')
#     st.markdown('NewScan은 경제뉴스에 대한 **긍정과 부정, 본문과 제목의 일치도, 낚시성 기사, 기사 요약** 등의 분석 정보'
#             '를 제공하여 경제 기사 소비자들이 보다 **편리하고 분석적인 시각**으로 뉴스 기사에 접근할 수 있도록 돕는 서비스입니다.')
with st.empty().container() :
    st.write('---')


_, col_logo, _ = st.columns([0.35,0.3,0.35])
with col_logo:
    st.image(Image.open('data/Logo.png'))


st.title('NewScan')
st.markdown('NewScan은 경제뉴스에 대한 \n'
            '**중립·긍정·부정의 감성라벨, 본문과 제목의 일치도, 낚시성 기사, 기사 요약** 등의 분석 정보를\n'
            ' 제공하여 경제 뉴스 소비자들이 보다 **편리하고 객관적인 시각**으로 뉴스에 접근할 수 있도록 돕는 서비스입니다.')

with st.empty().container() :
    st.write('---')

with st.expander('Service info', expanded=False) :
    st.markdown('')
    st.markdown("<h3 style='text-align: center;'>서비스 소개</h3>", unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')
    col_l, col_c, col_r = st.columns([0.05,0.9, 0.05])
    with col_c:
        st.markdown("<h5 style='font-weight : bold;'> 1.경제기사 검색 서비스<a href='/NEWSCAN' target='_self'> 🔗 </a></h5>", unsafe_allow_html=True)
        st.caption('일자, 키워드, 필터링을 통해 원하는 조건에 맞는 경제기사 분석 정보를 제공합니다.')
        col_1l, col_1r, col_2 = st.columns([0.15, 0.7, 0.15])
        with col_1r :
            st.image(Image.open('data/page_NEWSCAN_description.png'))
        # with col_1l:
        #     st.image(Image.open('data/page_NEWSCAN2.png'))
        # with col_2 :
            # st.markdown('')
            # st.markdown('**:three_button_mouse:이용방법**')
            # st.markdown('① 경제 기사를 확인하고 싶은 **일자**를 선택합니다.')
            # st.markdown('② 해당 일자에 언급빈도가 높은 순으로 정렬된 키워드 목록에서 원하는 **키워드**를 선택합니다.')
            # st.markdown('③ 원하는 신문사, 긍부정 감성라벨, 제목-본문일치율, 낚시기사 의심정도 등의 **필터값**을 선택합니다.')
            # st.markdown('④ 선택한 조건에 맞는 경제기사 건수의 중립, 긍정, 부정 비율을 보여줍니다.')
            # st.markdown('⑤ 선택한 조건에 맞는 경제기사 내에 언급빈도가 높은 인물, 장소, 기관 키워드가 색깔별로 분류되어 표시됩니다.')
            # st.markdown('⑥ 선택한 조건에 맞는 경제기사를 표시합니다.')
            # st.markdown('⑦ 원하는 기사 클릭시 하단에 긍부정 확률, 제목-본문 일치율, 피싱의심 확률, 기사 요약본, 원문 연결링크가 표시됩니다.')
            # st.markdown('⑧ 선택한 기사와 유사도가 높은 기사 10개를 보여줍니다.')

        st.markdown('-----')
        st.write('**:bulb: 분석방법**')
        st.write('(1) 워드클라우드 : **TfidfVectorizer**를 사용하여 인물, 장소, 기관별 상위 언급된 키워드 추출.')
        st.write('(2) 감성분석 : 한국언론진흥재단이 개발한 **KPFBERT** 분류모델을 사용하여 기사 본문의 중립, 긍정, 부정 라벨을 예측.')
        st.write('(3) 제목-본문 일치율 : **TF-IDF**로 토큰화한 제목과 본문 간의 **COS 유사도**를 확인하여 0.3미만은 불일치로 분류.')
        st.write('(4) 낚시성 기사 : **Bi-LSTM**을 사용하여 예측값이 0.5이상인 경우 피싱의심으로 분류.')
        st.write('(5) 기사 요약 : **Gensim**을 사용하여 추출적 요약으로 기사 요약.')


    st.markdown('-----')
    col_l, col_c2, col_r = st.columns([0.05, 0.9, 0.05])
    with col_c2:
        st.markdown('')
        st.markdown("<h5 style='font-weight : bold;'> 2.경제기사 키워드 리포트<a href='/REPORT' target='_self'> 🔗 </a></h5>",
                    unsafe_allow_html=True)
        st.caption('기간에 따른 키워드별 경제기사건수 변화, 긍정부정 기사건수 비율 변화 등의 정보를 시각화하여 제공합니다.')
        col_, col_3, col_ = st.columns([0.15, 0.7, 0.15])
        with col_3:
            st.image(Image.open('data/page_REPORT.png'))
