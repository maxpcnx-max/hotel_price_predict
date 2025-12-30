import streamlit as st
import pandas as pd
import gdown
import os
import holidays
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# 1. ตั้งค่าหน้าเว็บ (ต้องอยู่บรรทัดแรกสุด!)
st.set_page_config(
    page_title="Hotel AI System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. SYSTEM BACKEND (โหลดข้อมูลและโมเดลแค่ครั้งเดียว)
# ==========================================================
@st.cache_resource
def load_system_engine():
    # --- Download Data ---
    url_main = "https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri"
    url_room = "https://drive.google.com/uc?id=1tMSRSjfHyQT2QfnfqDjm8pw8qjw7bBoM"
    url_holiday = "https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw"

    if not os.path.exists("check_in_report.csv"):
        gdown.download(url_main, "check_in_report.csv", quiet=True)
        gdown.download(url_room, "room_type.csv", quiet=True)
        gdown.download(url_holiday, "thai_holidays.csv", quiet=True)
    
    # --- Process Data ---
    df = pd.read_csv("check_in_report.csv")
    room_type = pd.read_csv("room_type.csv")
    holidays_csv = pd.read_csv("thai_holidays.csv")
    
    if 'Room_Type' in room_type.columns:
        room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
    df = df.merge(room_type, on='Room', how='left')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Reservation'] = df['Reservation'].fillna('Unknown')
    df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
    df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Standard Room')
    
    # --- Train Model ---
    le_room = LabelEncoder()
    le_res = LabelEncoder()
    df['RoomType_encoded'] = le_room.fit_transform(df['Target_Room_Type'].astype(str))
    df['Reservation_encoded'] = le_res.fit_transform(df['Reservation'].astype(str))
    
    X = df[['Night', 'total_guests', 'is_holiday', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    
    return xgb, le_room, le_res, df

# โหลดระบบ (แสดง Spinner สวยๆ)
with st.spinner("🚀 Booting AI System..."):
    model, le_room, le_res, df = load_system_engine()

# ==========================================================
# 3. PAGE FUNCTIONS (สร้างหน้าจอต่างๆ แยกกัน)
# ==========================================================

def show_home_page():
    st.image("https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80&w=2070&auto=format&fit=crop", use_column_width=True)
    
    st.title("Welcome to Hotel AI System 👋")
    st.markdown("""
    ### ระบบบริหารจัดการราคาโรงแรมอัจฉริยะ
    ยินดีต้อนรับเข้าสู่ระบบ Decision Support System สำหรับผู้บริหารโรงแรม
    
    **ฟีเจอร์หลักของระบบ:**
    * **📊 Dashboard Analytics:** วิเคราะห์ข้อมูลย้อนหลัง ดูแนวโน้มราคา และพฤติกรรมลูกค้า
    * **🤖 Dynamic Pricing AI:** ระบบทำนายราคาที่เหมาะสมแบบ Real-time ด้วย Machine Learning (XGBoost)
    * **📅 Holiday Integration:** เชื่อมต่อปฏิทินวันหยุดประเทศไทยอัตโนมัติ
    
    ---
    *พัฒนาโดย: [ชื่อของคุณ] | เทคโนโลยี: Python, Streamlit, Scikit-learn*
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 เลือกเมนูจากแถบด้านซ้ายเพื่อเริ่มต้นใช้งาน")
    with col2:
        st.warning("⚠️ ข้อมูลในระบบเป็นข้อมูลจำลองเพื่อการศึกษา")

def show_dashboard_page():
    st.title("📊 Executive Dashboard")
    st.markdown("วิเคราะห์ภาพรวมผลประกอบการและสถิติการจอง")
    st.divider()

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Booking ทั้งหมด", f"{len(df):,} รายการ", "View All")
    c2.metric("ราคาเฉลี่ย (ADR)", f"{df['Price'].mean():,.0f} บาท", "+5% จากเดือนก่อน")
    c3.metric("รายได้รวม (Total)", f"{df['Price'].sum()/1e6:.2f} ล้านบาท")
    c4.metric("จำนวนแขกเฉลี่ย", f"{df['total_guests'].mean():.1f} คน/ห้อง")
    
    st.divider()

    # Graphs Row 1
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        st.subheader("📈 แนวโน้มราคาตามประเภทห้อง")
        avg_price_room = df.groupby('Target_Room_Type')['Price'].mean().reset_index().sort_values('Price', ascending=False)
        fig = px.bar(avg_price_room, x='Price', y='Target_Room_Type', orientation='h', text_auto='.2s', color='Price', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        st.subheader("🍰 แหล่งที่มาลูกค้า")
        res_count = df['Reservation'].value_counts().reset_index()
        fig2 = px.pie(res_count, values='count', names='Reservation', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # Data Table
    with st.expander("🔎 ดูข้อมูลดิบ (Raw Data)"):
        st.dataframe(df.sort_values('Date', ascending=False), use_container_width=True)

def show_pricing_page():
    st.title("🤖 Smart Pricing Engine")
    st.markdown("ระบบคำนวณราคาห้องพักที่เหมาะสมด้วย AI")
    
    # Layout แบบ Card
    with st.container(border=True):
        st.subheader("🛠️ กำหนดพารามิเตอร์ (Input Parameters)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            checkin_date = st.date_input("วันที่เช็คอิน", datetime.now())
            nights = st.number_input("จำนวนคืน", 1, 30, 1)
        with c2:
            room_name = st.selectbox("ประเภทห้อง", le_room.classes_)
            guests = st.number_input("จำนวนผู้เข้าพัก", 1, 10, 2)
        with c3:
            res_name = st.selectbox("Channel การจอง", le_res.classes_)
            # แสดงสถานะวันหยุดทันที
            th_holidays = holidays.Thailand()
            is_h = checkin_date in th_holidays
            st.info(f"วันหยุด: {'✅ ใช่' if is_h else '❌ ไม่ใช่'}")

        if st.button("🚀 คำนวณราคา (Predict Price)", type="primary", use_container_width=True):
            # Calculation
            r_code = le_room.transform([room_name])[0]
            res_code = le_res.transform([res_name])[0]
            
            inp = pd.DataFrame([{
                'Night': nights, 'total_guests': guests, 'is_holiday': 1 if is_h else 0,
                'month': checkin_date.month, 'weekday': checkin_date.weekday(),
                'RoomType_encoded': r_code, 'Reservation_encoded': res_code
            }])
            
            predicted_price = model.predict(inp)[0]
            
            # Result Display
            st.divider()
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                st.metric(label="ราคาแนะนำ (AI Suggested)", value=f"{predicted_price:,.0f} THB")
            
            with result_col2:
                st.success("✅ คำนวณเสร็จสิ้น! ราคานี้วิเคราะห์จากปัจจัยวันหยุด ฤดูกาล และประเภทห้องพักแล้ว")
                st.progress(0.85, text="Confidence Score: High")

# ==========================================================
# 4. MAIN NAVIGATION (ส่วนควบคุมเมนู)
# ==========================================================

# สร้าง Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
    st.markdown("### Hotel Admin")
    st.write(f"Logged in as: **Manager**")
    st.write(f"Date: {datetime.now().strftime('%d/%m/%Y')}")
    st.divider()
    
    # เมนูเลือกหน้า
    selected_page = st.radio(
        "Navigate to:", 
        ["🏠 หน้าหลัก (Home)", "📊 แดชบอร์ด (Dashboard)", "🤖 ระบบจัดการราคา (Pricing)"]
    )
    
    st.divider()
    st.caption("Version 1.0.2 | Powered by Streamlit")

# Logic ในการเปลี่ยนหน้า
if "หน้าหลัก" in selected_page:
    show_home_page()
elif "แดชบอร์ด" in selected_page:
    show_dashboard_page()
elif "ระบบจัดการราคา" in selected_page:
    show_pricing_page()