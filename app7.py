
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
import re

st.set_page_config(layout="wide")
st.title("📊 MZ세대 설문 데이터 대시보드")

@st.cache_data
def load_data():
    df_base = pd.read_csv("df.csv")
    df_cluster = pd.read_excel("cleaned_survey.xlsx")
    return df_base, df_cluster

df, df_cluster = load_data()
df.columns = df.columns.str.strip()

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

menu = st.sidebar.radio("📁 분석 메뉴", [
     "컬럼별 분포", "그룹별 분포", "클러스터링", "고객 페르소나", "인사이트 요약"
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
    st.markdown("## 👥 그룹별 분포 비교")

    group_col = st.selectbox("👥 그룹 기준 선택", df.columns)
    target_col = st.selectbox("📌 비교 대상 컬럼 선택", df.columns)

    is_group_multi = group_col in multi_col_choices
    is_target_multi = target_col in multi_col_choices

    if is_group_multi or is_target_multi:
        st.warning("복수응답 항목 포함됨: 선택 항목이 여러 개인 응답을 분해하여 집계합니다.")

        # 분해할 값 목록 추출
        group_values = multi_col_choices[group_col] if is_group_multi else df[group_col].dropna().unique()
        target_values = multi_col_choices[target_col] if is_target_multi else df[target_col].dropna().unique()

        grouped_data = {}

        for g in group_values:
            if is_group_multi:
                subset = df[df[group_col].apply(lambda x: g in str(x) if pd.notnull(x) else False)]
            else:
                subset = df[df[group_col] == g]

            counts = Counter()
            for val in subset[target_col].dropna():
                for t in target_values:
                    if t in str(val):
                        counts[t] += 1
            grouped_data[g] = counts

        grouped_df = pd.DataFrame(grouped_data).fillna(0).astype(int)

        # 상위 응답 기준 필터링
        top_items = grouped_df.sum(axis=1).sort_values(ascending=False).head(10).index
        fig = px.bar(grouped_df.loc[top_items].T, barmode='group',
                     title=f"{group_col}별 [{target_col}] 항목 분포 (복수 응답 포함)",
                     color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig, use_container_width=True)

        elif pd.api.types.is_numeric_dtype(df[target_col]):
        group_mean = df.groupby(group_col, as_index=False)[target_col].mean()  # ✅ 안전하게 처리
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
    st.markdown("- 사용 변수: 녹용 이미지, 녹용 매력요소, 녹용 구매장벽, 건기식 섭취 이유, 콘텐츠 행동, 건기식 릴스 호감요인")
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
        <tr><td><b>이름</b></td><td>루틴빌더 현정</td></tr>
        <tr><td><b>연령/성별</b></td><td>20~30대 초반 여성</td></tr>
        <tr><td><b>성향 키워드</b></td><td>실천 루틴 공유를 즐기는 MZ</td></tr>
        <tr><td><b>반응 특징</b></td><td>댓글, 좋아요, 저장 중심</td></tr>
        <tr><td><b>선호 콘텐츠</b></td><td>실제 사용 후기, 감각적 패키지, 루틴 연출</td></tr>
      </tbody>
    </table>
    """
    st.markdown(html_persona, unsafe_allow_html=True)

    # 콘텐츠 예시
    st.markdown("#### 콘텐츠 예시")
    st.markdown("- 5일 루틴 챌린지! 오늘도 한 포로 시작해요<br>- 이 언박싱, 진짜 감각 미쳤다 + 직장인 브이로그 연결", unsafe_allow_html=True)

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
          <td>댓글 유도 챌린지 / 후기 기반 숏폼 / 감각적 언박싱</td>
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
