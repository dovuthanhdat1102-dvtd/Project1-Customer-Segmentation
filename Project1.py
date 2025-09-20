import os
import base64
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from streamlit_option_menu import option_menu
from scipy.stats.mstats import winsorize
from sklearn.metrics import silhouette_score

import pickle
import streamlit as st
import squarify

# Hàm tiện ích
def check_columns(df, required_cols, context=""):
    """Kiểm tra các cột bắt buộc có trong DataFrame không"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Thiếu cột {missing} trong dữ liệu {context}. Vui lòng kiểm tra lại file input.")
        return False
    return True

# GUI setup
st.set_page_config(layout="wide")
st.title("Data Science & Machine Learning Project")
st.header("Customer Segmentation", divider='rainbow')

# Menu
with st.sidebar:
    choice = option_menu(
        "📊 Menu chính",
        ["Business Understanding", "Data Understanding", "Data Preparation", "Modeling & Evaluation", "Predict"],
        icons=["briefcase", "database", "gear", "bar-chart", "magic"],
        menu_icon="cast",
        default_index=0
    )

def load_data(uploaded_file, default_path=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.sidebar.success("File uploaded successfully!")
    elif default_path is not None and os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path, encoding='latin-1')
            st.sidebar.info(f"Dùng file mẫu: {os.path.basename(default_path)}")
        except Exception as e:
            st.error(f"❌ Không thể đọc file mẫu {default_path}: {e}")
            df = None
    else:
        df = None
    return df

def csv_download_link(df, csv_file_name, download_link_text):
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file_name}">{download_link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Initializing session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Business Understanding

if choice == 'Business Understanding':
    st.subheader("Business Objective")
    st.write("""
    ###### Phân khúc/nhóm/cụm khách hàng (market segmentation – còn được gọi là phân khúc thị trường) là quá trình nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung. Nó phân chia và nhóm khách hàng thành các nhóm nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm lý học, hành vi (geographic, demographic, psychographic, behavioral) và các đặc điểm khác.

    ###### Các nhà tiếp thị sử dụng kỹ thuật này để:
    - **Cá nhân hóa**: Điều chỉnh chiến lược tiếp thị để đáp ứng nhu cầu riêng của từng phân khúc.
    - **Tối ưu hóa**: Phân bổ hiệu quả nguồn lực tiếp thị.
    - **Insight**: Hiểu sâu hơn về cơ sở khách hàng.
    - **Tương tác**: Nâng cao sự tương tác và sự hài lòng của khách hàng.

    ###### => Vấn đề/Yêu cầu: Sử dụng Machine Learning và phân tích dữ liệu trong Python để thực hiện phân khúc khách hàng.
    """)
    st.image("RFM.png", caption="Customer Segmentation sử dụng RFM", use_container_width=True)

# Data Understanding

elif choice == 'Data Understanding':
    st.subheader("Upload hoặc chọn dữ liệu")

    # Upload dữ liệu
    st.markdown("**📦 Product File (thông tin sản phẩm)**")
    product_file = st.sidebar.file_uploader("Upload product file", type=['csv'], key="product")
    df_products = load_data(product_file, default_path="Data Project 1/Products_with_Categories.csv")

    st.markdown("**🛒 Transaction File (giao dịch)**")
    transaction_file = st.sidebar.file_uploader("Upload transaction file", type=['csv'], key="transaction")
    df_transactions = load_data(transaction_file, default_path="Data Project 1/Transactions.csv")

    if df_products is not None and df_transactions is not None:
        if "productId" not in df_transactions.columns:
            st.error("Không tìm thấy cột 'productId' trong transaction file.")
        else:
            df = pd.merge(df_transactions, df_products, on="productId", how="left")
            st.session_state['df'] = df

            # Tạo transactionId = Member_number + Date (coi mỗi KH trong 1 ngày là 1 giao dịch)
            df["transactionId"] = df["Member_number"].astype(str) + "_" + df["Date"].astype(str)

            st.write("### Data Overview")
            st.write("Số dòng:", df.shape[0])
            st.write("Số cột:", df.shape[1])

            # Tabs trong Data Understanding

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏰ Thời gian", "📂 Loại Hàng hoá", "🛍️ Sản phẩm", "🛍️ Sản phẩm đi kèm", "👤 Khách hàng"])

            with tab1:  # Thời gian
                st.markdown("### ⏰ Phân tích Doanh thu theo Thời gian")

                # Chuyển cột Date sang datetime
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

                # Thêm cột Month và Year
                df['Month'] = df['Date'].dt.to_period("M").astype(str)
                df['Year'] = df['Date'].dt.year

                # Bộ chọn mức phân tích
                view_option = st.selectbox(
                    "Chọn mức phân tích thời gian:",
                    ["Ngày", "Tháng", "Năm"]
                )

                if view_option == "Ngày":
                    daily_sales = df.groupby('Date')['price'].sum().reset_index()
                    st.markdown("#### 📅 Doanh thu theo Ngày")
                    st.line_chart(daily_sales.set_index("Date"))

                elif view_option == "Tháng":
                    monthly_sales = df.groupby('Month')['price'].sum().reset_index()
                    st.markdown("#### 📆 Doanh thu theo Tháng")
                    st.line_chart(monthly_sales.set_index("Month"))

                elif view_option == "Năm":
                    yearly_sales = df.groupby('Year')['price'].sum().reset_index()
                    st.markdown("#### 📊 Doanh thu theo Năm")
                    st.bar_chart(yearly_sales.set_index("Year"))


            with tab2:  # Loại hàng hoá
                st.markdown("### 📂 Phân tích theo Loại hàng hoá")

                if "Category" not in df.columns:
                    st.error("❌ Không tìm thấy cột 'Category' trong dữ liệu. Vui lòng kiểm tra lại file input.")
                else:
                    # 1️⃣ Doanh thu theo Category
                    revenue_by_cat = (
                        df.groupby("Category")["price"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    st.markdown("#### 💰 Top 10 Category theo Doanh thu")
                    st.bar_chart(revenue_by_cat.set_index("Category"))
                    st.dataframe(revenue_by_cat, use_container_width=True)

                    # 2️⃣ Số đơn hàng theo Category
                    orders_by_cat = (
                        df.groupby("Category")["productId"]
                        .count()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    orders_by_cat.rename(columns={"productId": "Orders"}, inplace=True)
                    st.markdown("#### 📦 Top 10 Category theo Số đơn hàng")
                    st.bar_chart(orders_by_cat.set_index("Category"))
                    st.dataframe(orders_by_cat, use_container_width=True)

                    # 3️⃣ Số khách mua theo Category
                    customers_by_cat = (
                        df.groupby("Category")["Member_number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    customers_by_cat.rename(columns={"Member_number": "Unique_Customers"}, inplace=True)
                    st.markdown("#### 👤 Top 10 Category theo Số khách hàng mua")
                    st.bar_chart(customers_by_cat.set_index("Category"))
                    st.dataframe(customers_by_cat, use_container_width=True)

            with tab3:  # Sản phẩm
                st.markdown("### 🛒 Phân tích theo Sản phẩm")

                if "productName" not in df.columns:
                    st.error("❌ Không tìm thấy cột 'productName' trong dữ liệu. Vui lòng kiểm tra lại file input.")
                else:
                    # 1️⃣ Doanh thu theo sản phẩm
                    revenue_by_prod = (
                        df.groupby("productName")["price"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    st.markdown("#### 💰 Top 10 Sản phẩm theo Doanh thu")
                    st.bar_chart(revenue_by_prod.set_index("productName"))
                    st.dataframe(revenue_by_prod, use_container_width=True)

                    # 2️⃣ Số đơn hàng theo sản phẩm
                    orders_by_prod = (
                        df.groupby("productName")["productId"]
                        .count()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    orders_by_prod.rename(columns={"productId": "Orders"}, inplace=True)
                    st.markdown("#### 📦 Top 10 Sản phẩm theo Số đơn hàng")
                    st.bar_chart(orders_by_prod.set_index("productName"))
                    st.dataframe(orders_by_prod, use_container_width=True)

                    # 3️⃣ Số khách mua theo sản phẩm
                    customers_by_prod = (
                        df.groupby("productName")["Member_number"]
                        .nunique()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    customers_by_prod.rename(columns={"Member_number": "Unique_Customers"}, inplace=True)
                    st.markdown("#### 👤 Top 10 Sản phẩm theo Số khách hàng mua")
                    st.bar_chart(customers_by_prod.set_index("productName"))
                    st.dataframe(customers_by_prod, use_container_width=True)

            with tab4: # TAB: Sản phẩm đi kèm
                st.subheader("🔗 Phân tích sản phẩm thường được mua kèm")

                # Đảm bảo có transactionId
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df['transactionId'] = df['Member_number'].astype(str) + "_" + df['Date'].dt.strftime('%Y-%m-%d')

                # Pivot: productName x user
                pivot = df.groupby(['productName','Member_number'])['items'].sum().unstack(fill_value=0)
                pivot_binary = (pivot > 0).astype(int)
                items = pivot_binary.index.tolist()
                item_to_idx = {it:i for i,it in enumerate(items)}

                # Cosine similarity (item-based CF)
                from sklearn.metrics.pairwise import cosine_similarity
                item_sim = cosine_similarity(pivot_binary.values)

                # Shrinkage theo số user chung
                co_matrix = pivot_binary @ pivot_binary.T  # co-occurrence counts (item x item)
                shrinkage_alpha = 5
                n_users = pivot_binary.shape[1]
                sim_shrinked = (co_matrix.values / n_users) / (
                    (co_matrix.values + shrinkage_alpha) / (n_users + shrinkage_alpha)
                )
                item_sim = item_sim * sim_shrinked  # combine raw sim + shrinkage

                # Co-occurrence counts cho fallback
                from collections import Counter
                baskets = df.groupby('transactionId')['productName'].apply(lambda x: set(x)).tolist()
                co_counts = {}
                for basket in baskets:
                    for a in basket:
                        co_counts.setdefault(a, Counter())
                        for b in basket:
                            if a == b: 
                                continue
                            co_counts[a][b] += 1

                # Apriori rules
                from mlxtend.frequent_patterns import apriori, association_rules
                from mlxtend.preprocessing import TransactionEncoder

                transactions = df.groupby('transactionId')['productName'].apply(list).tolist()
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                basket_df = pd.DataFrame(te_ary, columns=te.columns_)

                freq_items = apriori(basket_df, min_support=0.01, use_colnames=True)
                rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
                rules = rules[(rules['lift'] > 1.1) & (rules['confidence'] > 0.15)]

                # Function lấy đề xuất
                def get_combined_recs(prod, topn=10):
                    recs = []

                    # 1. Từ CF
                    if prod in item_to_idx:
                        sims = item_sim[item_to_idx[prod]]
                        idxs = sims.argsort()[::-1]
                        for j in idxs:
                            if items[j] == prod: 
                                continue
                            recs.append((items[j], float(sims[j]), 'CF'))

                    # 2. Từ Apriori
                    if not rules.empty:
                        related = rules[rules['antecedents'].apply(lambda x: prod in list(x))]
                        for _, row in related.iterrows():
                            for c in row['consequents']:
                                recs.append((c, float(row['confidence']), 'Apriori'))

                    # 3. Từ Co-occurrence fallback
                    co = co_counts.get(prod, Counter())
                    for p, cnt in co.most_common(50):
                        recs.append((p, float(cnt), 'Co-occurrence'))

                    # Nếu hoàn toàn rỗng -> return []
                    if not recs:
                        return pd.DataFrame(columns=['product','score','source'])

                    # Gộp kết quả
                    dfc = pd.DataFrame(recs, columns=['product','score','source'])
                    dfc = dfc.groupby('product').agg({'score':'max','source':'first'}).reset_index()
                    dfc['score_norm'] = (dfc['score'] - dfc['score'].min()) / (dfc['score'].max() - dfc['score'].min() + 1e-9)
                    dfc = dfc.sort_values('score_norm', ascending=False).head(topn)
                    return dfc

                # UI
                selected = st.selectbox("Chọn sản phẩm để phân tích", sorted(df['productName'].unique()))

                if st.button("Xem đề xuất mua kèm"):
                    rec_df = get_combined_recs(selected, topn=10)

                    if rec_df.empty:
                        st.warning("❌ Không tìm thấy sản phẩm mua kèm. Hãy thử sản phẩm khác.")
                    else:
                        st.write("📊 Đề xuất sản phẩm thường mua kèm với:", selected)
                        st.dataframe(rec_df[['product','score_norm','source']])
                        st.bar_chart(rec_df.set_index('product')['score_norm'])

            with tab5:  # Khách hàng
                st.markdown("### 👥 Phân tích theo Khách hàng")

                if "Member_number" not in df.columns or "price" not in df.columns:
                    st.error("❌ Thiếu cột 'Member_number' hoặc 'price' trong dữ liệu.")
                else:
                    # Chuẩn hóa dữ liệu ngày
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

                    # Tổng số lần mua của khách hàng
                    purchase_count = (
                        df.groupby("Member_number")["productId"]
                        .count()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    purchase_count.rename(columns={"productId": "Total_Purchases"}, inplace=True)
                    st.markdown("#### 🛒 Top 10 khách hàng theo Tổng số lần mua")
                    st.bar_chart(purchase_count.set_index("Member_number"))
                    st.dataframe(purchase_count, use_container_width=True)

                    # Tổng chi tiêu của khách hàng
                    total_spent = (
                        df.groupby("Member_number")["price"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    total_spent.rename(columns={"price": "Total_Spent"}, inplace=True)
                    st.markdown("#### 💰 Top 10 khách hàng theo Tổng chi tiêu")
                    st.bar_chart(total_spent.set_index("Member_number"))
                    st.dataframe(total_spent, use_container_width=True)

                    # Trung bình chi tiêu mỗi đơn
                    avg_spent = (
                        df.groupby("Member_number")["price"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    avg_spent.rename(columns={"price": "Avg_Spent_per_Order"}, inplace=True)
                    st.markdown("#### 📊 Top 10 khách hàng theo Trung bình chi tiêu mỗi đơn")
                    st.bar_chart(avg_spent.set_index("Member_number"))
                    st.dataframe(avg_spent, use_container_width=True)

                    # Ngày mua gần nhất của khách hàng
                    last_purchase = (
                        df.groupby("Member_number")["Date"]
                        .max()
                        .reset_index()
                    )
                    last_purchase = last_purchase.sort_values("Date", ascending=False).head(10)
                    last_purchase.rename(columns={"Date": "Last_Purchase_Date"}, inplace=True)
                    st.markdown("#### 📅 Top 10 khách hàng có ngày mua gần nhất")
                    st.dataframe(last_purchase, use_container_width=True)

            # Cho phép tải CSV đã merge
            csv_download_link(df, "merged_data.csv", "📥 Download merged CSV")
    else:
        st.warning("⚠️ Vui lòng upload hoặc kiểm tra file mẫu trong thư mục `Data Project 1`")

if choice == 'Data Preparation':
    st.subheader("Data Preparation & EDA")

    if st.session_state['df'] is not None:
        df = st.session_state['df'].copy()

        if not check_columns(df, ["Date", "items", "price", "Member_number"], context="Transactions"):
            st.stop()
        # Đảm bảo Date về dạng datetime
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df['Day'] = df['Date'].dt.to_period("D").dt.to_timestamp()

        # Tạo biến Sales
        df['Sales'] = df['items'] * df['price']

        st.markdown("### 📊 Data Summary")
        st.write(df.describe())

        # Missing values
        st.markdown("### 🕳 Missing Values")
        st.write(df.isna().sum())

        # Duplicate check
        dup_cnt = df.duplicated(subset=["Member_number", "Date", "price"]).sum()
        st.write(f"**Likely duplicates**: {dup_cnt} rows (subset on Member_number, Date, price)")

        # Histograms
        st.markdown("### Distribution of Price & log(1+Price)")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(df["price"], bins=40, alpha=.8)
        ax[0].set_title("Histogram price")
        ax[1].hist(np.log1p(df["price"]), bins=40, alpha=.8)
        ax[1].set_title("Histogram log(1+price)")
        st.pyplot(fig)

        # Daily revenue
        st.markdown("### 📈 Daily Revenue with MA7")
        daily = df.groupby("Day")["price"].sum().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily.index, daily.values, alpha=.4, label="daily")
        ax.plot(daily.index, daily.rolling(7, min_periods=1).mean(), label="7d MA")
        ax.set_title("Daily revenue")
        ax.legend()
        st.pyplot(fig)

        # Monthly revenue
        st.markdown("### 📅 Monthly Revenue")
        df["month"] = df["Date"].dt.to_period("M").astype(str)
        rev_mon = df.groupby("month")["price"].sum()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rev_mon.index, rev_mon.values, marker="o")
        ax.set_title("Monthly revenue")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Deduplication
        dup_mask = df.duplicated(subset=["Member_number","Date","price"], keep="first")
        dup_n = int(dup_mask.sum()); dup_pct = dup_n / max(len(df), 1)
        st.markdown("### 🔎 Duplicate Check")
        st.write(f"- Số dòng trùng: {dup_n} / {len(df):,} ({dup_pct:.2%})")
        if dup_n > 0:
            st.write("- Ví dụ trùng lặp:")
            st.dataframe(df.loc[dup_mask].head(10))
        df_dedup = df.drop_duplicates(subset=["Member_number","Date","price"]).copy()
        st.write(f"- Sau drop_duplicates: {len(df_dedup):,} dòng (giảm {dup_n:,})")

        # DAILY COLLAPSE

        st.markdown("## 🗓 Daily Collapse (customer-day level)")
        df_daily = df.groupby(['Member_number', 'Day']).agg(
            Sales=('Sales', 'sum')
        ).reset_index()

        st.write(f"Sau khi gộp: {len(df):,} dòng → {len(df_daily):,} dòng")
        st.dataframe(df_daily.head())

        # Lưu lại df_daily để Modeling dùng
        st.session_state['df_daily'] = df_daily

        # RFM Calculation

        st.markdown("## 🧮 RFM Calculation (sau khi gộp)")
        max_date = df_daily['Day'].max()

        # Recency
        recency = df_daily.groupby('Member_number')["Day"].max().reset_index()
        recency['Recency'] = (max_date - recency['Day']).dt.days

        # Frequency
        frequency = df_daily.groupby('Member_number')["Day"].nunique().reset_index()
        frequency.columns = ['Member_number', 'Frequency']

        # Monetary
        monetary = df_daily.groupby('Member_number')["Sales"].sum().reset_index()
        monetary.columns = ['Member_number', 'Monetary']

        # Merge RFM
        rfm = recency[['Member_number', 'Recency']] \
                .merge(frequency, on='Member_number') \
                .merge(monetary, on='Member_number')

        st.write("### RFM Table (raw)")
        st.dataframe(rfm.head())

        # Winsorize + Scaling
        st.markdown("## ⚖️ Winsorize + Scaling")
        rfm_scaled = rfm.copy()
        for col in ["Recency", "Frequency", "Monetary"]:
            rfm_scaled[col] = np.array(winsorize(rfm_scaled[col], limits=[0.01, 0.01]))
        
        robust = RobustScaler()
        rfm_scaled[["Recency", "Frequency", "Monetary"]] = robust.fit_transform(
            rfm_scaled[["Recency", "Frequency", "Monetary"]]
        )
        mm = MinMaxScaler()
        rfm_scaled[["Recency", "Frequency", "Monetary"]] = mm.fit_transform(
            rfm_scaled[["Recency", "Frequency", "Monetary"]]
        )

        st.write("### RFM Table (sau Winsorize + Scaling)")
        st.dataframe(rfm_scaled.head())

        # Lưu vào session_state
        st.session_state['rfm'] = rfm
        st.session_state['rfm_scaled'] = rfm_scaled

        st.success("✅ RFM đã được tính toán và scale thành công. Sẵn sàng cho bước Modeling!")

    else:
        st.warning("⚠️ Bạn cần chuẩn bị dữ liệu ở bước Data Understanding trước.")


# Modeling & Evaluation     

elif choice == 'Modeling & Evaluation':
    st.write("### Modeling With KMeans")

    if 'df_daily' in st.session_state and st.session_state['df_daily'] is not None:
        df_daily = st.session_state['df_daily'].copy()

        # 1. Tính RFM
        recent_date = df_daily['Day'].max()
        df_RFM = df_daily.groupby('Member_number').agg({
            'Day': lambda x: (recent_date - x.max()).days,
            'Member_number': 'count',
            'Sales': 'sum'
        }).rename(columns={
            'Day': 'Recency',
            'Member_number': 'Frequency',
            'Sales': 'Monetary'
        }).reset_index()

        st.subheader("📊 RFM Table (sau khi Daily Collapse)")
        st.dataframe(df_RFM.head())

        # 2. Tiền xử lý (log, winsorize, scaling)
        X = df_RFM[['Recency', 'Frequency', 'Monetary']].copy()
        for col in X.columns:
            X[col] = np.log1p(X[col])
            X[col] = np.array(winsorize(X[col], limits=[0.01, 0.01]))

        robust = RobustScaler()
        X_scaled = robust.fit_transform(X)
        mm = MinMaxScaler()
        X_final = mm.fit_transform(X_scaled)

        # 3. Elbow method
        sse = {}
        for k in range(1, 15):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_final)
            sse[k] = kmeans.inertia_

        fig, ax = plt.subplots()
        ax.set_title('Phương pháp Elbow')
        ax.set_xlabel('Số cụm (k)')
        ax.set_ylabel('Tổng Bình phương khoảng cách')
        sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
        st.pyplot(fig)

        # 4. Chọn số cluster
        n_clusters = st.sidebar.number_input(
            'Chọn số lượng cụm k (2-10):',
            min_value=2, max_value=10, value=4, step=1
        )
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X_final)

        df_RFM['Cluster'] = model.labels_

        # 5. Thống kê theo cluster
        cluster_stats = df_RFM.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).round(2)
        cluster_stats.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        cluster_stats['Percent'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(2)
        cluster_stats.reset_index(inplace=True)

        # 6. Gán Segment
        def assign_segment(row, df_RFM):
            R_med, F_med, M_med = (
                df_RFM['Recency'].median(),
                df_RFM['Frequency'].median(),
                df_RFM['Monetary'].median()
            )
            R_q75 = df_RFM['Recency'].quantile(0.75)

            if row['RecencyMean'] < R_med and row['FrequencyMean'] > F_med and row['MonetaryMean'] > M_med:
                return "Champions"
            elif row['FrequencyMean'] > F_med and row['MonetaryMean'] > M_med:
                return "Loyal Customers"
            elif row['RecencyMean'] < R_med and row['FrequencyMean'] <= F_med:
                return "Potential Loyalist"
            elif row['MonetaryMean'] > M_med and row['FrequencyMean'] <= F_med:
                return "Big Spenders"
            elif row['RecencyMean'] > R_q75 and row['MonetaryMean'] > M_med:
                return "At Risk"
            elif row['RecencyMean'] > R_q75 and row['MonetaryMean'] <= M_med:
                return "Hibernating"
            else:
                return "Lost"

        cluster_stats['Segment'] = cluster_stats.apply(assign_segment, axis=1, df_RFM=df_RFM)

        st.subheader("📈 Thống kê theo cụm + Segment")
        st.dataframe(cluster_stats)

        # 7. Bảng giải thích Segment
        st.subheader("📖 Giải thích Segment & Chiến lược kinh doanh")
        segment_explain = pd.DataFrame([
            ["Champions", "Khách mới mua gần đây, mua nhiều, chi tiêu cao", "Chăm sóc VIP, ưu đãi đặc biệt, early access sản phẩm"],
            ["Loyal Customers", "Khách trung thành, mua thường xuyên, giá trị cao", "Duy trì loyalty program, upsell, xây cộng đồng"],
            ["Potential Loyalist", "Khách mới, có tiềm năng trở thành trung thành", "Tặng voucher, remarketing khuyến khích quay lại"],
            ["Big Spenders", "Mua ít nhưng chi tiêu lớn", "Upsell, cross-sell, gợi ý sp cao cấp"],
            ["At Risk", "Từng mua nhiều nhưng đang ít quay lại", "Re-engagement: email, khuyến mãi quay lại"],
            ["Hibernating", "Khách cũ, lâu rồi không quay lại, chi tiêu thấp", "Remarketing nhẹ nhàng, không đầu tư quá nhiều"],
            ["Lost", "Khách đã rời bỏ, ít giá trị", "Chỉ tiếp cận khi có campaign lớn"]
        ], columns=["Segment", "Ý nghĩa", "Chiến lược kinh doanh"])
        st.dataframe(segment_explain)

        # 8. Visualization
        fig_scatter = px.scatter(
            cluster_stats, x='RecencyMean', y='MonetaryMean',
            size='FrequencyMean', color='Segment', size_max=60
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_3d = px.scatter_3d(
            cluster_stats, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
            color='Segment', size='Count'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("🗺️ TreeMap phân khúc khách hàng")
        fig_treemap, ax_treemap = plt.subplots()
        fig_treemap.set_size_inches(14, 10)
        squarify.plot(
            sizes=cluster_stats['Count'],
            label=[
                f"{row.Segment}\n"
                f"Recency: {row.RecencyMean:.1f}\n"
                f"Freq: {row.FrequencyMean:.1f}\n"
                f"Monetary: {row.MonetaryMean:.1f}\n"
                f"{row.Count} KH ({row.Percent}%)"
                for _, row in cluster_stats.iterrows()
            ],
            color=sns.color_palette("tab10", len(cluster_stats)).as_hex(),
            alpha=0.7,
            text_kwargs={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax_treemap.set_title("Phân khúc khách hàng (TreeMap)", fontsize=22, fontweight="bold")
        ax_treemap.axis('off')
        st.pyplot(fig_treemap)

        # 9. Export Model
        if st.button("💾 Xuất mô hình"):
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump((model, cluster_stats, robust, mm), f)
            st.session_state['model_exported'] = True
            st.success("✅ Mô hình đã được huấn luyện và lưu thành công! Bây giờ bạn có thể sang tab Predict để dự đoán.")

    else:
        st.warning("❌ Không có dữ liệu. Vui lòng chạy bước Data Preparation trước.")


# Predict

elif choice == 'Predict':
    st.subheader("Dự đoán Cụm cho Khách hàng mới")

    if 'model_exported' in st.session_state and st.session_state['model_exported']:
        # Load lại mô hình và cluster_stats
        with open('kmeans_model.pkl', 'rb') as f:
            model, cluster_stats, robust, mm = pickle.load(f)

        st.write("### 📊 Thống kê cụm hiện tại")
        st.dataframe(cluster_stats)

        # 1. Upload file KH mới
        uploaded_predict_file = st.file_uploader("📂 Upload file khách hàng (CSV/Excel)", type=["csv", "xlsx"])
        if 'df_new' not in st.session_state:
            st.session_state['df_new'] = pd.DataFrame(columns=['Member_number', 'Day', 'Sales'])

        if uploaded_predict_file is not None:
            try:
                if uploaded_predict_file.name.endswith(".csv"):
                    df_uploaded = pd.read_csv(uploaded_predict_file)
                else:
                    df_uploaded = pd.read_excel(uploaded_predict_file)

                required_cols = ['Member_number', 'Day', 'Sales']
                if all(col in df_uploaded.columns for col in required_cols):
                    df_uploaded['Day'] = pd.to_datetime(df_uploaded['Day'])
                    st.session_state['df_new'] = pd.concat([st.session_state['df_new'], df_uploaded], ignore_index=True)
                    st.success("✅ Đã thêm dữ liệu từ file upload!")
                else:
                    st.error(f"❌ File phải có đủ các cột: {required_cols}")
            except Exception as e:
                st.error(f"⚠️ Lỗi khi đọc file: {e}")

        # 2. Nhập dữ liệu khách hàng mới
        member_number = st.text_input("🆔 Mã Khách hàng:")
        last_purchase = st.date_input("📅 Ngày mua gần nhất:")
        sales = st.number_input("💰 Số tiền giao dịch:", min_value=0.0)

        if st.button("➕ Thêm khách hàng"):
            new_data = pd.DataFrame({
                'Member_number': [member_number],
                'Day': [pd.to_datetime(last_purchase)],
                'Sales': [sales]
            })
            st.session_state['df_new'] = pd.concat([st.session_state['df_new'], new_data], ignore_index=True)

        st.write("### 📥 Dữ liệu khách hàng đã thêm")
        st.dataframe(st.session_state['df_new'])

        # 3. Dự đoán
        if st.button("🚀 Dự đoán"):
            if len(st.session_state['df_new']) == 0:
                st.warning("⚠️ Bạn cần nhập ít nhất 1 khách hàng để dự đoán.")
            else:
                df_new = st.session_state['df_new'].copy()

                # Tính RFM cho dữ liệu mới
                max_date = pd.Timestamp.now().normalize()
                recency = df_new.groupby('Member_number')["Day"].max().reset_index()
                recency['Recency'] = (max_date - recency['Day']).dt.days

                frequency = df_new.groupby('Member_number')["Day"].nunique().reset_index()
                frequency.columns = ['Member_number', 'Frequency']

                monetary = df_new.groupby('Member_number')["Sales"].sum().reset_index()
                monetary.columns = ['Member_number', 'Monetary']

                df_RFM_new = recency[['Member_number', 'Recency']] \
                                .merge(frequency, on='Member_number') \
                                .merge(monetary, on='Member_number')

                st.write("### 🧮 Bảng RFM cho KH mới")
                st.dataframe(df_RFM_new)

                # Winsorize + Scale
                X_new = df_RFM_new[['Recency', 'Frequency', 'Monetary']].copy()
                for col in X_new.columns:
                    X_new[col] = np.array(winsorize(X_new[col], limits=[0.01, 0.01]))
                X_new = robust.transform(X_new)
                X_new = mm.transform(X_new)

                # Dự đoán Cluster
                cluster_pred = model.predict(X_new)
                df_RFM_new['Cluster'] = cluster_pred

                # Gán Segment dựa trên cluster_stats
                cluster_map = cluster_stats.set_index("Cluster")["Segment"].to_dict()
                df_RFM_new['Segment'] = df_RFM_new['Cluster'].map(cluster_map)

                # Gán chiến lược kinh doanh
                strategy_map = {
                    "Champions": "Chăm sóc VIP, ưu đãi đặc biệt, early access sản phẩm",
                    "Loyal Customers": "Duy trì loyalty program, upsell, xây cộng đồng",
                    "Potential Loyalist": "Tặng voucher, remarketing khuyến khích quay lại",
                    "Big Spenders": "Upsell, cross-sell, gợi ý sp cao cấp",
                    "At Risk": "Re-engagement: email, khuyến mãi quay lại",
                    "Hibernating": "Remarketing nhẹ nhàng, không đầu tư quá nhiều",
                    "Lost": "Chỉ tiếp cận khi có campaign lớn"
                }
                df_RFM_new['Chiến lược'] = df_RFM_new['Segment'].map(strategy_map)

                # Hiển thị kết quả
                st.success("✅ Dự đoán thành công!")
                st.write("### 📌 Kết quả dự đoán")
                st.dataframe(df_RFM_new)

                # Cho phép tải kết quả
                csv_download_link(df_RFM_new, 'RFM_prediction_results.csv', '📥 Tải xuống kết quả')

    else:
        st.warning("❌ Bạn phải xuất mô hình trước khi dự đoán.")

    # User Feedback

    st.write("### 💬 User Feedback")
    user_feedback = st.text_area("Chia sẻ ý kiến của bạn:", value='')

    if st.button("📩 Gửi Feedback"):
        current_time = datetime.now()
        feedback_df = pd.DataFrame({
            'Time': [current_time],
            'Feedback': [user_feedback]
        })

        if not os.path.isfile('feedback.csv'):
            feedback_df.to_csv('feedback.csv', index=False)
        else:
            feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)

        st.success("👍 Cảm ơn bạn đã gửi phản hồi!")





