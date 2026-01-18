import streamlit as st
from PIL import Image
import numpy as np
import os, sqlite3, json
import pandas as pd
import gdown
from ultralytics import YOLO
from datetime import datetime

# -------------------------
# CONFIGURATION
# -------------------------

st.set_page_config(page_title="GreenLoop", page_icon="♻️", layout="wide")

DB_PATH = "data.db"
IMG_DIR = "uploads"
os.makedirs(IMG_DIR, exist_ok=True)

ORG_CREDENTIALS = {
    "ngo@example.org": "password123",
    "bbmp@example.gov": "bbmp_pass"
}

GDRIVE_ID = "1Y_uW_GrpJthpJwHcW_0nk8eszy-a_lBN"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
MODEL_PATH = "best.pt"

# -------------------------
# CUSTOM CSS (DARK/LIGHT MODE COMPATIBLE)
# -------------------------

def inject_css():
    st.markdown("""
        <style>
            /* This ensures the green colors look good in both modes */
            :root {
                --brand-green: #2e7d32;
                --accent-green: #4caf50;
            }
            
            /* Target Headers specifically */
            h1, h2, h3 {
                color: var(--brand-green) !important;
            }

            /* Buttons styling */
            .stButton button {
                background-color: var(--brand-green);
                color: white !important;
                border: none;
                border-radius: 8px;
            }
            
            .stButton button:hover {
                background-color: var(--accent-green);
                color: white !important;
            }

            /* Customizing cards for requests */
            .request-card {
                padding: 1.5rem;
                border-radius: 10px;
                border: 1px solid #4caf50;
                margin-bottom: 1rem;
            }
            
            /* Footer Styling */
            .footer {
                text-align: center;
                color: #888;
                margin-top: 50px;
                font-size: 0.8rem;
            }
        </style>
    """, unsafe_allow_html=True)

inject_css()

# -------------------------
# DATABASE SETUP
# -------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY,
            user_email TEXT,
            user_name TEXT,
            location TEXT,
            image_path TEXT,
            counts_json TEXT,
            timestamp TEXT,
            accepted INTEGER DEFAULT 0,
            accepted_by TEXT,
            accepted_time TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# -------------------------
# MODEL LOADING
# -------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model…"):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error("Failed to download model. Please check the URL.")
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------
# SESSION INITIALIZATION
# -------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.user_email = None
    st.session_state.user_name = None

# -------------------------
# AUTHENTICATION
# -------------------------

def login_page():
    st.markdown("## 🔐 Welcome to GreenLoop")
    st.caption("Empowering communities for smarter waste management.")
    user_type = st.radio("I am a:", ["Common User", "Organization"])
    
    if user_type == "Common User":
        name = st.text_input("Name")
        email = st.text_input("Email")
        if st.button("Login as User"):
            if name and email:
                st.session_state.logged_in = True
                st.session_state.user_type = "user"
                st.session_state.user_email = email
                st.session_state.user_name = name
                st.rerun()
            else:
                st.error("Please enter both name and email.")
    else:
        email = st.text_input("Organization Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Login as Org"):
            if ORG_CREDENTIALS.get(email) == pwd:
                st.session_state.logged_in = True
                st.session_state.user_type = "org"
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid credentials.")

def logout():
    for key in ["logged_in", "user_type", "user_email", "user_name"]:
        st.session_state.pop(key, None)

# -------------------------
# DATABASE FUNCTIONS
# -------------------------

def add_request(user_email, user_name, location, image_path, counts):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c = conn.cursor()
    c.execute("""
        INSERT INTO requests
          (user_email,user_name,location,image_path,counts_json,timestamp)
        VALUES (?,?,?,?,?,?)
    """, (user_email, user_name, location, image_path, json.dumps(counts), ts))
    conn.commit()

def get_user_requests(email):
    return pd.read_sql("SELECT * FROM requests WHERE user_email = ? ORDER BY id DESC", conn, params=(email,))

def get_pending_requests():
    return pd.read_sql("SELECT * FROM requests WHERE accepted = 0 ORDER BY id DESC", conn)

def accept_request(req_id, org_email):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c = conn.cursor()
    c.execute("""
        UPDATE requests
        SET accepted = 1, accepted_by = ?, accepted_time = ?
        WHERE id = ?
    """, (org_email, ts, req_id))
    conn.commit()

def notify_user(email, subject, message):
    # Mock notification
    st.toast(f"Notification sent to {email}")
    print(f"[NOTIFY] To: {email}\nSubject: {subject}\n{message}")

# -------------------------
# USER PAGES
# -------------------------

def user_upload_page():
    st.markdown("## ♻️ Smart Waste Reporting")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded = st.file_uploader("Upload Dumpster Image", type=["jpg", "jpeg", "png"])
        loc = st.text_input("Location Address")
        
    if uploaded and loc:
        image = Image.open(uploaded).convert("RGB")
        with col2:
            st.image(image, caption="Preview", use_container_width=True)
            
        if st.button("Analyze & Report"):
            img_np = np.array(image)
            with st.spinner("🔍 AI is analyzing waste types..."):
                res = model(img_np)[0]
                classes = res.boxes.cls.cpu().numpy().astype(int)
                names = model.names
                counts = {}
                for c_ in classes:
                    lbl = names[c_]
                    counts[lbl] = counts.get(lbl, 0) + 1
                ann = res.plot()
            
            st.image(ann, caption="Detection Result", use_container_width=True)
            
            # Save data
            fname = f"{int(datetime.now().timestamp())}_{uploaded.name}"
            path = os.path.join(IMG_DIR, fname)
            Image.fromarray(ann).save(path)
            
            add_request(st.session_state.user_email,
                        st.session_state.user_name,
                        loc, path, counts)

            st.success("✅ Success! Your request has been logged.")
            
            # Email Notification
            notify_user("ngo@example.org", 
                        "New Waste Pickup Requested", 
                        f"Location: {loc} reported by {st.session_state.user_name}")

def user_history_page():
    st.markdown("## 📋 My History")
    df = get_user_requests(st.session_state.user_email)
    
    if df.empty:
        st.info("You haven't submitted any reports yet.")
        return
        
    for _, row in df.iterrows():
        with st.container():
            status_color = "green" if row.accepted else "orange"
            st.markdown(f"""
            <div style="border-left: 5px solid {status_color}; padding-left: 15px; margin-bottom: 20px;">
                <h4>Request #{row.id} - {row.location}</h4>
                <p><b>Date:</b> {row.timestamp}</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                if os.path.exists(row.image_path):
                    st.image(row.image_path, width=200)
            with c2:
                counts = json.loads(row.counts_json)
                st.write("**Detected Items:**")
                st.write(", ".join([f"{k}: {v}" for k, v in counts.items()]))
                if row.accepted:
                    st.success(f"Picked up by {row.accepted_by} at {row.accepted_time}")
                else:
                    st.warning("Status: Pending Pickup")
            st.divider()

# -------------------------
# ORG DASHBOARD
# -------------------------

def org_dashboard_page():
    st.markdown("## 🏢 Management Dashboard")
    df = get_pending_requests()
    
    tab1, tab2 = st.tabs(["Pending Tasks", "Export Data"])
    
    with tab1:
        if df.empty:
            st.write("No pending waste collection requests! 🎉")
        else:
            for _, row in df.iterrows():
                with st.expander(f"📍 {row.location} (Requested by {row.user_name})"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(row.image_path, use_container_width=True)
                    with col2:
                        st.write(f"**Reported on:** {row.timestamp}")
                        counts = json.loads(row.counts_json)
                        st.write("**Waste Composition:**")
                        for k, v in counts.items():
                            st.write(f"- {k}: {v}")
                        
                        if st.button(f"Mark as Collected", key=f"btn_{row.id}"):
                            accept_request(row.id, st.session_state.user_email)
                            notify_user(row.user_email, "Waste Collected", "Your waste report has been addressed.")
                            st.rerun()

    with tab2:
        all_data = pd.read_sql("SELECT * FROM requests", conn)
        st.dataframe(all_data)
        csv = all_data.to_csv(index=False).encode()
        st.download_button("📥 Download Master Log", csv, "greenloop_data.csv", "text/csv")

# -------------------------
# MAIN FLOW
# -------------------------

if not st.session_state.logged_in:
    login_page()
else:
    # Sidebar Navigation
    st.sidebar.title("🌱 GreenLoop")
    st.sidebar.markdown(f"**Welcome, {st.session_state.user_email}**")
    
    if st.session_state.user_type == "user":
        menu = ["Report Waste", "My History"]
    else:
        menu = ["Dashboard"]
        
    choice = st.sidebar.radio("Navigation", menu)
    
    if st.sidebar.button("Log Out"):
        logout()
        st.rerun()
        
    if choice == "Report Waste":
        user_upload_page()
    elif choice == "My History":
        user_history_page()
    elif choice == "Dashboard":
        org_dashboard_page()

# -------------------------
# FOOTER
# -------------------------

st.markdown(f"""
    <div class="footer">
        <hr>
        <p>2025 GreenLoop. Developed by Spandana A P, Shravya P, Surbhi Sneha, Sridevi Shetty.</p>
    </div>
""", unsafe_allow_html=True)
