
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
import re

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from os import path

FONT_PATH = "C:/Users/aaa01/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf"

st.set_page_config(layout="wide")
st.title("📊 MZ세대 설문 데이터 대시보드")

@st.cache_data
def load_data():
    df_base = pd.read_csv("df.csv")
    df_cluster = pd.read_excel("cleaned_survey.xlsx")
    df_text = pd.read_csv("df2.csv", encoding="utf-8")
    return df_base, df_cluster

df, df_cluster = load_data()
df.columns = df.columns.str.strip()
df_text.columns = df_text.columns.str.strip()

multi_col_choices = {
    '건기식_섭취제품': ['비타민', '유산균', '홍삼', '오메가3', '단백질 파우더', '루테인', '마그네슘', '콜라겐'],
    '녹용_이미지': ['고급 건강식품이다', '떠오르는 이미지가 없다', '맛이 부담스럽다', '면역력에 좋다',
                '부모님/어르신용이다', '부정적이다(제품, 효능에 대한 의심, 사슴 불쌍하다)', '비싸다', '효과가 좋다'],
    '녹용_구매장벽': ['가격 부담', '동물복지이슈', '맛/향에 대한 거부감', '복용 방법이 번거로움', '부작용',
                 '젊은 세대와 어울리지 않는 이미지', '정보 부족', '제품 품질에 대한 의심', '효능에 대한 의심'],
    '녹용_매력요소': ['과학적 효능 인증 & 원산지 투명성', '맛 개선(부담 없는 맛)', '스틱형/젤리형/양갱 등 간편한 섭취 방식',
                 '유명 인플루언서/SNS 바이럴', '트렌디한 디자인과 패키지', '합리적인 가격대'],
    '콘텐츠_행동': ['공유', '단순 시청(아무 액션 X)', '댓글 작성', '저장', '좋아요', '팔로우'],
    '건기식릴스_호감요인': ['감각적인 비주얼/디자인일 때', '신뢰할 만한 정보가 포함될 때', '신뢰할 수 있고 사회적 책임을 다하는 브랜드라고 느껴질 때',
                      '실제 사용자가 등장할 때', '이벤트/할인 정보가 있을 때', '재미있고 트렌디할 때',
                      '제품 효능이 눈에 띄게 전달될 때', '호감이 안생김'],
    '건기식릴스_기억키워드': ['간편함', '건강 루틴', '고객 후기', '없음', '이벤트/특가', '천연 재료',
                       '효능 (면역력, 체력 회복, 혈액 순환 등)']
}
def explode_counts_safe(series, known_choices):
    counter = Counter()
    for val in series.dropna():
        for choice in known_choices:
            if choice in str(val):
                counter[choice] += 1
    return pd.Series(counter).sort_values(ascending=False)

def split_responses(text):
    if not isinstance(text, str):
        return []
    brackets = re.findall(r'\(.*?\)', text)
    placeholders = [f"#BRACKET{i}#" for i in range(len(brackets))]
    temp_text = text
    for b, p in zip(brackets, placeholders):
        temp_text = temp_text.replace(b, p)
    parts = [part.strip() for part in temp_text.split(",")]
    final_parts = []
    for part in parts:
        for p, b in zip(placeholders, brackets):
            part = part.replace(p, b)
        final_parts.append(part)
    return final_parts

# ✅ 클러스터링용 전처리 수행 (C 시나리오)
c_df = df_cluster[["녹용_이미지", "녹용_매력요소", "녹용_구매장벽", "건기식_섭취이유", "콘텐츠_행동", "건기식릴스_호감요인"]].copy()
for col in c_df.columns:
    c_df[col] = c_df[col].fillna("").apply(split_responses)

encoded_parts = []
for col in c_df.columns:
    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(mlb.fit_transform(c_df[col]), columns=[f"{col}_{cls}" for cls in mlb.classes_])
    encoded_parts.append(encoded)

c_encoded_fixed = pd.concat(encoded_parts, axis=1)
c_scaled_fixed = StandardScaler().fit_transform(c_encoded_fixed)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(c_scaled_fixed)
df_cluster['cluster'] = labels

# ✅ 사이드바 메뉴
menu = st.sidebar.radio("📁 분석 메뉴", [
    "컬럼별 분포", "그룹별 분포", "클러스터링", "고객 페르소나", "텍스트 분석", "인사이트 요약"
])

if menu == "컬럼별 분포":
    col = st.selectbox("📈 분포를 보고 싶은 컬럼 선택", df.columns)
    if col in multi_col_choices:
        st.warning("복수응답 항목입니다.")
        counts = explode_counts_safe(df[col], multi_col_choices[col]).reset_index()
        counts.columns = ['항목', '응답 수']
        fig = px.bar(counts, x='응답 수', y='항목', orientation='h', title=f"[{col}] 항목별 응답 수")
    else:
        counts = df[col].value_counts().sort_values(ascending=True).reset_index()
        counts.columns = ['항목', '응답 수']
        fig = px.bar(counts, x='응답 수', y='항목', orientation='h', title=f"[{col}] 응답 분포")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "그룹별 분포":
    group_col = st.selectbox("👥 그룹 기준 선택", df.select_dtypes(include='object').columns)
    target_col = st.selectbox("📌 비교 대상 컬럼 선택", df.columns)

    if target_col in multi_col_choices:
        grouped_data = {}
        for g in df[group_col].dropna().unique():
            subset = df[df[group_col] == g]
            grouped_data[g] = explode_counts_safe(subset[target_col], multi_col_choices[target_col])
        grouped_df = pd.DataFrame(grouped_data).fillna(0).astype(int)
        top_items = grouped_df.sum(axis=1).sort_values(ascending=False).head(10).index
        fig = px.bar(grouped_df.loc[top_items].T, barmode='group', title=f"{group_col}별 [{target_col}] 항목 분포")
        st.plotly_chart(fig, use_container_width=True)
    elif pd.api.types.is_numeric_dtype(df[target_col]):
        group_mean = df.groupby(group_col)[target_col].mean().reset_index()
        fig = px.bar(group_mean, x=group_col, y=target_col, title=f"{group_col}별 {target_col} 평균",
                     color=group_col, color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig, use_container_width=True)
    else:
        cross = pd.crosstab(df[group_col], df[target_col])
        cross_percent = cross.div(cross.sum(axis=1), axis=0)
        fig = px.bar(cross_percent, barmode='stack', title=f"{group_col}별 {target_col} 비율")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "클러스터링":
    st.subheader("✅변수 설명")
    st.markdown("- 사용 변수: 녹용 이미지, 매력요소, 구매장벽, 건기식 섭취 이유, 콘텐츠 행동, 릴스 호감요인")
    st.markdown("- 선택 이유: 제품에 대한 태도 + 소비자 콘텐츠 반응 + 실제 섭취 행동이 모두 반영되어 있음")

    st.subheader("📉 PCA 시각화 (클러스터 분포)")
    pca = PCA(n_components=2)
    components = pca.fit_transform(c_scaled_fixed)
    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
    df_pca["cluster"] = labels
    fig = px.scatter(df_pca, x="PC1", y="PC2", color=df_pca["cluster"].astype(str), title="PCA 시각화")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 클러스터별 인구통계 분포")
    for cat_col in ["연령대", "성별", "직업"]:
        st.markdown(f"#### ▪ {cat_col} 기준")
        fig = px.histogram(df_cluster, x=cat_col, color="cluster", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 클러스터 요약 통계 (유의미 항목만)")

    # 표 데이터 준비
    summary_df = c_encoded_fixed.copy()
    summary_df["cluster"] = labels
    cluster_summary = summary_df.groupby("cluster").mean()

    # 유의미 컬럼 추출 로직 수정
    overall_mean = cluster_summary.mean()
    std_dev = cluster_summary.std()
    deviation = (cluster_summary - overall_mean).abs()
    key_columns = deviation.loc[:, deviation.gt(std_dev).any(axis=0)].columns

    # 필터링 및 강조
    cluster_summary_filtered = cluster_summary[key_columns].T.round(2)
    df = cluster_summary_filtered.copy()

    # HTML 테이블 구성
    def render_html_table(df):
        html = "<style>th, td { padding: 6px 10px; text-align: center; }</style>"
        html += "<table border='1' style='border-collapse:collapse;'>"
        html += "<tr style='background-color:#f0f2f6;'>"
        html += "<th>Cluster</th>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"

        for idx, row in df.iterrows():
            html += f"<tr><td><b>{idx}</b></td>"
            for col in df.columns:
                val = row[col]
                if isinstance(val, (float, int)):
                    is_max = val == df[col].max()
                    style = "font-weight:bold; color:#2b6cb0;" if is_max else ""
                    html += f"<td style='{style}'>{val:.2f}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        html += "</table>"
        return html

    st.markdown(render_html_table(cluster_summary_filtered), unsafe_allow_html=True)

    st.markdown("""
    #### 🔍 수치 기반 클러스터 해석 요약

    - **Cluster 0**: 효능에 대한 관심은 있으나 SNS 반응성 낮음 → 정보 기반 관망형
    - **Cluster 1**: 가격/신뢰 중시, 콘텐츠는 소비하나 반응 낮음 → 정보탐색형
    - **Cluster 2**: 댓글·좋아요·저장 모두 높음 → SNS 릴스 실천형 타겟
    """)



elif menu == "고객 페르소나":
    st.header("🧍 고객 페르소나")

    # 클러스터 비교 표
    st.markdown("#### 📋 클러스터 요약 비교 (초기 해석 vs 릴스 기준 재해석)")

    html_compare = """
    <table style='width:100%; border-collapse:collapse; text-align:center;'>
      <thead style='background-color:#f0f2f6;'>
        <tr>
          <th>클러스터</th>
          <th>초기 해석</th>
          <th>재해석</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>Cluster 0</b></td>
          <td>부모님용 인식, 콘텐츠 반응 낮음</td>
          <td style='color:gray;'>릴스 반응 거의 없음 → SNS 콘텐츠 타겟 아님</td>
        </tr>
        <tr>
          <td><b>Cluster 1</b></td>
          <td>가격 부담 큼, 정보 기반 신뢰 중시</td>
          <td style='color:gray;'>콘텐츠 소비하나 반응 낮음 → 정보 콘텐츠 적합</td>
        </tr>
        <tr>
          <td><b style='color:#2b6cb0;'>Cluster 2 ✅</b></td>
          <td>댓글/좋아요 활발, 트렌디 콘텐츠 민감</td>
          <td style='color:#2b6cb0; font-weight:bold;'>릴스 저장/댓글/좋아요 높음 → 핵심 릴스 타겟</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(html_compare, unsafe_allow_html=True)

    # 페르소나 카드
    st.markdown("#### 👤 Cluster 2 페르소나 요약")

    html_persona = """
    <table style='width:80%; border-collapse:collapse; text-align:center;'>
      <thead style='background-color:#f9f9f9;'>
        <tr><th>항목</th><th>내용</th></tr>
      </thead>
      <tbody>
        <tr><td><b>연령/성별</b></td><td>20~30대 초반 여성</td></tr>
        <tr><td><b>성향 키워드</b></td><td>건강루틴 & 공유를 즐기는 & 피로회복이 필요한 MZ</td></tr>
        <tr><td><b>반응 특징</b></td><td>댓글, 좋아요, 저장 중심의 참여</td></tr>
        <tr><td><b>선호 콘텐츠</b></td><td>실제 사용 후기, 트렌디한, 일상 루틴</td></tr>
      </tbody>
    </table>
    """
    st.markdown(html_persona, unsafe_allow_html=True)

    # 콘텐츠 예시
    st.markdown("#### 콘텐츠 예시")
    st.markdown("- 5일 루틴 브이로그! 매일 가벼운 한 포로 시작해요<br> - 간편하게 피로회복 + 직장인 브이로그 연결", unsafe_allow_html=True)

    # 전략 표
    st.markdown("#### 📊 클러스터별 콘텐츠 전략 매핑")

    html_strategy = """
    <table style='width:100%; border-collapse:collapse; text-align:center;'>
      <thead style='background-color:#f0f2f6;'>
        <tr>
          <th>클러스터</th>
          <th>전략 키워드</th>
          <th>톤/포맷 예시</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b style='color:#2b6cb0;'>Cluster 2 ✅</b></td>
          <td>SNS 루틴형, 참여 유도 중심</td>
          <td>댓글 유도 챌린지 / 실제 후기 기반 숏폼 / 패키징 언박싱</td>
        </tr>
        <tr>
          <td><b>Cluster 1</b></td>
          <td>정보성 콘텐츠, 비교 중심</td>
          <td>전문가 코멘트, 혜택 설명형 콘텐츠</td>
        </tr>
        <tr>
          <td><b>Cluster 0</b></td>
          <td>감성/신뢰 기반, 선물 포지셔닝</td>
          <td>카카오 선물하기, 감성 브랜딩, 엄마 선물템 강조</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(html_strategy, unsafe_allow_html=True)

    st.markdown("#### 종합")
    st.markdown("- 핵심 타겟은 <b style='color:#2b6cb0;'>Cluster 2 – 참여형 MZ 루틴 소비자</b>", unsafe_allow_html=True)

elif menu == "텍스트 분석":
    submenu = st.sidebar.radio("📑 텍스트 분석 세부 메뉴", ["전체 키워드 분석", "유형별 키워드 분석"])

    def render_html_table(df, title="빈도표"):
        max_val = df.select_dtypes(include='number').max().max()
        html = f"<h5>{title}</h5>"
        html += "<style>th, td { padding: 6px 10px; text-align: center; font-family: NanumGothic; } th { background-color: #f0f2f6; }</style>"
        html += "<table border='1' style='border-collapse:collapse; width:100%'>"
        html += "<tr>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
        for _, row in df.iterrows():
            html += "<tr>"
            for val in row:
                if isinstance(val, (int, float)):
                    blue_shade = int(255 - (val / max_val) * 100)
                    html += f"<td style='background-color:rgb({blue_shade},{blue_shade+25},255);'>{val:,}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    def run_analysis(df, top_n=10, qcat=None):
        # 빈도 분석
        freq = df['카테고리'].value_counts().reset_index()
        freq.columns = ['카테고리', '빈도']
        st.subheader(f"📊 TOP {top_n} 키워드 빈도")

        col1, col2 = st.columns([1, 1])
        with col1:
            word_freq = df['카테고리'].value_counts().head(top_n).to_dict()
            wc = WordCloud(max_font_size=30, max_words=top_n, background_color='white', relative_scaling=.5,
                          width=500, height=250, font_path=FONT_PATH).generate_from_frequencies(word_freq)
            fig_wc, ax = plt.subplots(figsize=(5, 2.5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)

        with col2:
            render_html_table(freq.head(top_n))

        st.markdown("""
        🔍 **인사이트 요약**
        - 상위 키워드는 '가격', '효능', '신뢰', '간편성' 등 반복적으로 등장
        - 이는 MZ세대가 건기식 선택 시 중시하는 가치 반영
        - 예: "비싸다" → 가격 장벽 해소 콘텐츠, "효능" → 과학적인 효능 설명, "간편한 제형" -> 젤리형 숏폼 강조
        """)

        # 토픽 분석
        st.subheader("🧠 자주 등장하는 키워드 조합")
        docs = df.groupby("ID")["카테고리"].apply(list)
        topic_counter = Counter()
        for cats in docs:
            unique = list(set(cats))
            if len(unique) >= 2:
                topic_counter.update(itertools.combinations(sorted(unique), 2))
        top_topics = [(a[0], a[1], count) for a, count in topic_counter.most_common(top_n)]
        if top_topics:
            topic_df = pd.DataFrame(top_topics, columns=['키워드1', '키워드2', '동시 등장 수'])
            render_html_table(topic_df, title=f"키워드 조합 TOP {top_n}")

            st.markdown("""
            🔍 **인사이트 요약**
            - '가격 장벽' + '효능 불확실' 조합 다수 등장 → 가격 변동은 제약 사항
            - 고객은 단일 요인보다 **복합 조건** 충족 시 구매 고려
            - '정보/후기 + 효능/성분' or '트렌디 + 가벼운 제형' 조합 콘텐츠가 효과적
            """)
        else:
            st.info("충분한 데이터가 없어 조합 분석을 생략합니다.")

        # 연관 분석
        st.subheader("🔗 연관 키워드 분석")
        pair_counter = Counter()
        grouped = df.groupby("ID")["카테고리"].apply(list)
        for keywords in grouped:
            pairs = itertools.combinations(sorted(set(keywords)), 2)
            pair_counter.update(pairs)
        top_pairs = [(a[0], a[1], count) for a, count in pair_counter.most_common(top_n)]
        if top_pairs:
            top_df = pd.DataFrame(top_pairs, columns=['키워드1', '키워드2', '빈도'])
            render_html_table(top_df, title=f"연관 키워드 TOP {top_n}")

            st.markdown("""
            🔍 **인사이트 요약**
            - 예: '효능 불확실' ↔ '정보 부족', '실제 후기' ↔ '실제 후기'
            - 실제 사용 후기 & 과학적 설명 중심 콘텐츠 필요
            """)
        else:
            st.info("충분한 데이터가 없어 연관 분석을 생략합니다.")

        # 질문 카테고리별 인사이트 요약
        if qcat:
            if qcat == "건기식 장벽 요인":
                st.markdown("""
                🔍 **인사이트 요약**
                - '가격 부담', '효능 의심'이 주된 이유
                - 고객은 가성비와 확실한 효과를 중시함
                - 가격 대비 효과를 강조한 실제 사용자 리뷰 콘텐츠 필요
                """)
            elif qcat == "녹용 장벽 요인":
                st.markdown("""
                🔍 **인사이트 요약**
                - '어르신용 이미지', '복용 번거로움'이 장벽
                - 젤리/스틱형 등 간편 포맷, 패키징 개선 콘텐츠 필요
                """)
            elif qcat == "sns콘텐츠 호감 유형":
                st.markdown("""
                🔍 **인사이트 요약**
                - '실제 사용자', '친근한 분위기' 선호
                - 광고 느낌을 줄이고 공감 가능한 스토리 중심 콘텐츠 기획 필요
                """)
            elif qcat == "sns콘텐츠 장벽 유형":
                st.markdown("""
                🔍 **인사이트 요약**
                - '과장된 연출', '가짜 같은 표현'에 반감 큼
                - 인위적인 모델 연출 대신 사용 후기 기반 콘텐츠 권장
                """)
            elif qcat == "건기식 호감 요인":
                st.markdown("""
                🔍 **인사이트 요약**
                - 브랜드 신뢰성, 과학적 정보, 트렌디한 이미지에 긍정 반응
                - 이 3가지 요소를 조합한 브랜드 스토리텔링 콘텐츠 기획
                """)
            elif qcat == "건강관리 필요 요인":
                st.markdown("""
                🔍 **인사이트 요약**
                - 체력 저하, 피로 누적 등의 시점에 건강기능식품 관심 증가
                - '지쳤을 때 필요한 한 포' 메시지로 캠페인 구성 적합
                """)
            elif qcat == "녹용 이미지":
                st.markdown("""
                🔍 **인사이트 요약**
                - '어르신용', '비싸다', '맛이 부담스럽다' 등 인식 존재
                - 이미지 전환 위한 젊은 감성의 언박싱 콘텐츠 필요
                """)

        st.markdown("""
        📌 콘텐츠 방향 예시
        | 인사이트 유형 | 콘텐츠 전략 예시 |
        |----------------|------------------|
        | 부정 키워드 | '왜 효과 없다고 느꼈나요?' -> 실제 후기/리뷰 인터뷰 |
        | 긍정 키워드 | '간편한 제형 좋아요' -> 젤리스틱형 릴스 콘텐츠 |
        | 연관 패턴 | '트렌디한 디자인 + 신뢰성' 강조 숏폼 -> 스토리텔링형 리뷰 |
        """)

        st.markdown("""
        🧪 A/B 테스트 콘텐츠 제안
        - **A안:** "과학으로 입증된 녹용 효과" 전문가 등장형 콘텐츠
        - **B안:** "20대가 선택한 하루 한 포 루틴템" 사용자 후기 기반 콘텐츠
        """)
        
    if submenu == "전체 키워드 분석":
        st.header("📚 전체 카테고리 기반 분석")
        run_analysis(df_text, top_n=10)

    elif submenu == "유형별 키워드 분석":
        st.header("📚 유형별 키워드 분석")
        selected_qcat = st.selectbox("질문 카테고리 선택", df_text['질문 카테고리'].dropna().unique())
        sub_df = df_text[df_text['질문 카테고리'] == selected_qcat]
        run_analysis(sub_df, top_n=5, qcat=selected_qcat)
