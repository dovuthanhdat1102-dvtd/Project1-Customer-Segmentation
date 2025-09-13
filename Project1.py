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

import pickle
import streamlit as st
import squarify

# GUI setup
st.set_page_config(layout="wide")
st.title("Data Science & Machine Learning Project")
st.header("Customer Segmentation", divider='rainbow')

# Menu 
menu = ["Business Understanding", "Data Understanding", "Data Preparation", "Modeling & Evaluation", "Predict"]
choice = st.sidebar.selectbox('Menu', menu)

def load_data(uploaded_file, default_path=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.sidebar.success("File uploaded successfully!")
    elif default_path is not None and os.path.exists(default_path):
        df = pd.read_csv(default_path, encoding='latin-1')
        st.sidebar.info(f"Dùng file mẫu: {os.path.basename(default_path)}")
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

# -------------------------
# Business Understanding
# -------------------------
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

# -------------------------
# Data Understanding
# -------------------------
elif choice == 'Data Understanding':
    st.subheader("Upload hoặc chọn dữ liệu")

    st.markdown("**📦 Product File (thông tin sản phẩm)**")
    product_file = st.sidebar.file_uploader("Upload product file", type=['csv'], key="product")
    df_products = load_data(product_file, default_path="Data Project 1/Products_with_Categories.csv")

    st.markdown("**🛒 Transaction File (giao dịch)**")
    transaction_file = st.sidebar.file_uploader("Upload transaction file", type=['csv'], key="transaction")
    df_transactions = load_data(transaction_file, default_path="Data Project 1/Transactions.csv")

    if df_products is not None and df_transactions is not None:
        # nếu cột tên khác, bạn có thể sửa mapping trước khi merge
        # đảm bảo productId tồn tại ở cả 2
        if "productId" not in df_transactions.columns:
            st.error("Không tìm thấy cột 'productId' trong transaction file.")
        else:
            df = pd.merge(df_transactions, df_products, on="productId", how="left")
            st.session_state['df'] = df

            st.write("### Data Overview")
            st.write("Number of rows:", df.shape[0])
            st.write("Number of columns:", df.shape[1])
            st.write("First five rows of the merged data:")
            st.write(df.head())

            csv_download_link(df, "merged_data.csv", "📥 Download merged CSV")
    else:
        st.warning("⚠️ Vui lòng upload hoặc kiểm tra file mẫu trong thư mục `Data Project 1`")

from scipy.stats.mstats import winsorize

# Data Preparation
# -------------------------
if choice == 'Data Preparation':
    st.subheader("Data Preparation & EDA")

    if st.session_state['df'] is not None:
        df = st.session_state['df'].copy()

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

        # =====================
        # DAILY COLLAPSE
        # =====================
        st.markdown("## 🗓 Daily Collapse (customer-day level)")
        df_daily = df.groupby(['Member_number', 'Day']).agg(
            Sales=('Sales', 'sum')
        ).reset_index()

        st.write(f"Sau khi gộp: {len(df):,} dòng → {len(df_daily):,} dòng")
        st.dataframe(df_daily.head())

        # Lưu lại df_daily để Modeling dùng
        st.session_state['df_daily'] = df_daily

        # =====================
        # RFM Calculation
        # =====================
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
            rfm_scaled[col] = winsorize(rfm_scaled[col], limits=[0.01, 0.01])

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
# -------------------------
elif choice == 'Modeling & Evaluation':
    st.write("### Modeling With KMeans")

    if 'df_daily' in st.session_state and st.session_state['df_daily'] is not None:
        df_daily = st.session_state['df_daily'].copy()

        # =========================
        # 1. Tính RFM từ df_daily
        # =========================
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

        # =========================
        # 2. WINSORIZE + SCALING
        # =========================
        X = df_RFM[['Recency', 'Frequency', 'Monetary']].copy()

        # Winsorize trước
        for col in X.columns:
            X[col] = winsorize(X[col], limits=[0.01, 0.01])

        # RobustScaler
        robust = RobustScaler()
        X_scaled = robust.fit_transform(X)

        # MinMaxScaler
        mm = MinMaxScaler()
        X_final = mm.fit_transform(X_scaled)

        # =========================
        # 3. ELBOW METHOD
        # =========================
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

        # =========================
        # 4. FIT KMEANS
        # =========================
        n_clusters = st.sidebar.number_input(
            'Chọn số lượng cụm k (2-10):',
            min_value=2, max_value=10, value=3, step=1
        )
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X_final)

        df_RFM['Cluster'] = model.labels_

        # =========================
        # 5. CLUSTER STATS
        # =========================
        cluster_stats = df_RFM.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).round(2)
        cluster_stats.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        cluster_stats['Percent'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(2)
        cluster_stats.reset_index(inplace=True)
        cluster_stats['Cluster'] = 'Cụm ' + cluster_stats['Cluster'].astype(str)

        st.subheader("📈 Thống kê theo cụm")
        st.dataframe(cluster_stats)

        # =========================
        # 6. NÚT EXPORT MODEL
        # =========================
        if st.button("💾 Xuất mô hình"):
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump((model, cluster_stats, robust, mm), f)
            st.session_state['model_exported'] = True
            st.success("✅ Mô hình đã được huấn luyện và lưu thành công! Bây giờ bạn có thể sang tab Predict để dự đoán.")

        # =========================
        # 7. VISUALIZATION
        # =========================
        fig_scatter = px.scatter(
            cluster_stats, x='RecencyMean', y='MonetaryMean',
            size='FrequencyMean', color='Cluster', size_max=60
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_3d = px.scatter_3d(
            cluster_stats, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
            color='Cluster', size='Count'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("🗺️ TreeMap phân khúc khách hàng")
        colors_dict = {
            0: 'green', 1: 'red', 2: 'royalblue',
            3: 'orange', 4: 'purple', 5: 'pink',
            6: 'brown', 7: 'cyan', 8: 'yellow', 9: 'gray'
        }

        fig_treemap, ax_treemap = plt.subplots()
        fig_treemap.set_size_inches(14, 10)
        squarify.plot(
            sizes=cluster_stats['Count'],
            label=[
                f"{row.Cluster}\n"
                f"Recency: {row.RecencyMean:.1f}\n"
                f"Freq: {row.FrequencyMean:.1f}\n"
                f"Monetary: {row.MonetaryMean:.1f}\n"
                f"{row.Count} KH ({row.Percent}%)"
                for _, row in cluster_stats.iterrows()
            ],
            color=[colors_dict.get(i, 'gray') for i in range(len(cluster_stats))],
            alpha=0.7,
            text_kwargs={'fontsize': 12, 'fontweight': 'bold'}
        )
        ax_treemap.set_title("Phân khúc khách hàng (TreeMap)", fontsize=22, fontweight="bold")
        ax_treemap.axis('off')
        st.pyplot(fig_treemap)

    else:
        st.warning("❌ Không có dữ liệu. Vui lòng chạy bước Data Preparation trước.")

# Predict
# -------------------------
elif choice == 'Predict':
    st.subheader("🔮 Dự đoán Cụm cho Khách hàng mới")

    if 'model_exported' in st.session_state and st.session_state['model_exported']:
        # Load lại mô hình và cluster_stats
        with open('kmeans_model.pkl', 'rb') as f:
            model, cluster_stats, robust, mm = pickle.load(f)

        st.write("### 📊 Thống kê cụm hiện tại")
        st.dataframe(cluster_stats)

        # Nhập dữ liệu khách hàng mới
        member_number = st.text_input("🆔 Mã Khách hàng:")
        last_purchase = st.date_input("📅 Ngày mua gần nhất:")
        sales = st.number_input("💰 Số tiền giao dịch:", min_value=0.0)

        # Lưu dữ liệu nhập vào session_state
        if 'df_new' not in st.session_state:
            st.session_state['df_new'] = pd.DataFrame(columns=['Member_number', 'Day', 'Sales'])

        if st.button("➕ Thêm khách hàng"):
            new_data = pd.DataFrame({
                'Member_number': [member_number],
                'Day': [pd.to_datetime(last_purchase)],
                'Sales': [sales]
            })
            st.session_state['df_new'] = pd.concat([st.session_state['df_new'], new_data], ignore_index=True)

        st.write("### 📥 Dữ liệu khách hàng đã thêm")
        st.dataframe(st.session_state['df_new'])

        # Nút dự đoán
        if st.button("🚀 Dự đoán"):
            if len(st.session_state['df_new']) == 0:
                st.warning("⚠️ Bạn cần nhập ít nhất 1 khách hàng để dự đoán.")
            else:
                df_new = st.session_state['df_new'].copy()

                # =========================
                # 1. Tính RFM cho dữ liệu mới
                # =========================
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

                # =========================
                # 2. Winsorize + Scale dữ liệu mới
                # =========================
                X_new = df_RFM_new[['Recency', 'Frequency', 'Monetary']].copy()

                # Winsorize như lúc train
                for col in X_new.columns:
                    X_new[col] = winsorize(X_new[col], limits=[0.01, 0.01])

                # Scale bằng RobustScaler + MinMaxScaler đã fit
                X_new = robust.transform(X_new)
                X_new = mm.transform(X_new)

                # =========================
                # 3. Dự đoán
                # =========================
                cluster_pred = model.predict(X_new)
                df_RFM_new['Cluster'] = cluster_pred

                st.success("✅ Dự đoán thành công!")
                st.write("### 📌 Kết quả dự đoán")
                st.dataframe(df_RFM_new)

                # Cho phép tải kết quả
                csv_download_link(df_RFM_new, 'RFM_prediction_results.csv', '📥 Tải xuống kết quả')

    else:
        st.warning("❌ Bạn phải xuất mô hình trước khi dự đoán.")

    # =====================
    # User Feedback
    # =====================
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





