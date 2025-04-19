# app.py

import streamlit as st
from PIL import Image
import numpy as np
import os, sqlite3, json, io
import pandas as pd
import gdown
from ultralytics import YOLO
from datetime import datetime

### --- CONFIG & DATABASE SETUP --- ###

DB_PATH = "data.db"
IMG_DIR = "uploads"
os.makedirs(IMG_DIR, exist_ok=True)

ORG_CREDENTIALS = {
    "ngo@example.org": "password123",   # demo org account
    "bbmp@example.gov": "bbmp_pass"     # demo BBMP account
}

GDRIVE_ID = "1Y_uW_GrpJthpJwHcW_0nk8eszy-a_lBN"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"
MODEL_PATH = "best.pt"

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

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model…"):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

### --- AUTHENTICATION & SESSION --- ###

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.user_email = None
    st.session_state.user_name = None

def login():
    st.title("🔐 Login")
    user_type = st.radio("I am a:", ["Common User", "Organization"])
    if user_type == "Common User":
        name = st.text_input("Name")
        email = st.text_input("Email")
        if st.button("Login"):
            if name and email:
                st.session_state.update({
                    "logged_in": True,
                    "user_type": "user",
                    "user_email": email,
                    "user_name": name
                })
                st.experimental_rerun()
            else:
                st.error("Please enter both name and email.")
    else:
        email = st.text_input("Organization Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if ORG_CREDENTIALS.get(email) == pwd:
                st.session_state.update({
                    "logged_in": True,
                    "user_type": "org",
                    "user_email": email
                })
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")

def logout():
    for k in ["logged_in","user_type","user_email","user_name"]:
        st.session_state.pop(k, None)
    st.experimental_rerun()

### --- DATABASE HELPERS --- ###

def add_request(user_email, user_name, location, image_path, counts):
    ts = datetime.now().isoformat()
    c = conn.cursor()
    c.execute("""
        INSERT INTO requests
          (user_email,user_name,location,image_path,counts_json,timestamp)
        VALUES (?,?,?,?,?,?)
    """, (user_email, user_name, location, image_path, json.dumps(counts), ts))
    conn.commit()

def get_user_requests(email):
    df = pd.read_sql("SELECT * FROM requests WHERE user_email = ?", conn, params=(email,))
    return df

def get_pending_requests():
    df = pd.read_sql("SELECT * FROM requests WHERE accepted = 0", conn)
    return df

def accept_request(req_id, org_email):
    ts = datetime.now().isoformat()
    c = conn.cursor()
    c.execute("""
        UPDATE requests
        SET accepted = 1, accepted_by = ?, accepted_time = ?
        WHERE id = ?
    """, (org_email, ts, req_id))
    conn.commit()

### --- EMAIL / NOTIFICATION PLACEHOLDER --- ###

def notify_user(email, subject, message):
    # In real life integrate SMTP or push notification here.
    print(f"[NOTIFY] To: {email}\nSubject: {subject}\n{message}")

### --- PAGES FOR COMMON USER --- ###

def user_upload_page():
    st.header("🖼️ Upload Dumpster Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    loc = st.text_input("Location of dumpster")
    if uploaded and loc:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Preview", use_column_width=True)
        if st.button("Run Detection"):
            img_np = np.array(image)
            with st.spinner("🔍 Detecting…"):
                res = model(img_np)[0]
                classes = res.boxes.cls.cpu().numpy().astype(int)
                names = model.names
                counts = {}
                for c_ in classes:
                    lbl = names[c_]
                    counts[lbl] = counts.get(lbl,0) + 1
                ann = res.plot()
            st.image(ann, caption="Result", use_column_width=True)
            # save image & DB
            fname = f"{datetime.now().timestamp():.0f}_{uploaded.name}"
            path = os.path.join(IMG_DIR, fname)
            Image.fromarray(ann).save(path)
            add_request(st.session_state.user_email,
                        st.session_state.user_name,
                        loc, path, counts)
            st.success("✅ Saved to your requests.")
            # download CSV report
            df = pd.DataFrame([{
                "location": loc,
                **counts,
                "timestamp": datetime.now().isoformat()
            }])
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download report (CSV)", csv, "report.csv")
            # notify NGO/BBMP
            if st.button("🔔 Notify BBMP/NGO"):
                notify_user("ngo@example.org",
                            "New dumpster upload",
                            f"{st.session_state.user_name} uploaded at {loc}.")
                st.info("Notification sent to BBMP/NGO.")

def user_history_page():
    st.header("📋 My Requests")
    df = get_user_requests(st.session_state.user_email)
    if df.empty:
        st.info("No requests yet.")
        return
    for _, row in df.iterrows():
        st.markdown(f"**Request #{row.id}** — {row.timestamp}")
        st.markdown(f"- Location: {row.location}")
        counts = json.loads(row.counts_json)
        for k,v in counts.items():
            st.markdown(f"  - {k}: {v}")
        status = "✅ Accepted" if row.accepted else "⏳ Pending"
        st.markdown(f"- Status: **{status}**")
        if row.accepted:
            st.markdown(f"  - by: {row.accepted_by} on {row.accepted_time}")
        # download per-request CSV
        rep = pd.DataFrame([{
            "location": row.location,
            **counts,
            "timestamp": row.timestamp,
            "accepted": row.accepted,
            "accepted_by": row.accepted_by or "",
            "accepted_time": row.accepted_time or ""
        }])
        csv = rep.to_csv(index=False).encode()
        st.download_button(f"Download #{row.id}", csv, f"req_{row.id}.csv")
        st.markdown("---")

### --- ORGANIZATION DASHBOARD --- ###

def org_dashboard_page():
    st.header("📊 Organization Dashboard")
    df = get_pending_requests()
    if df.empty:
        st.info("No pending requests.")
        return
    for _, row in df.iterrows():
        st.markdown(f"**Request #{row.id}** — by {row.user_name} ({row.user_email})")
        st.markdown(f"- Location: {row.location}")
        image = Image.open(row.image_path)
        st.image(image, width=300)
        counts = json.loads(row.counts_json)
        for k,v in counts.items():
            st.markdown(f"  - {k}: {v}")
        if st.button(f"Accept #{row.id}", key=f"acc{row.id}"):
            accept_request(row.id, st.session_state.user_email)
            notify_user(row.user_email,
                        "Your dumpster request has been accepted",
                        f"Your request at {row.location} will be collected by {st.session_state.user_email}.")
            st.success(f"Request #{row.id} marked accepted.")
        st.markdown("---")
    # CSV export of all (including accepted)
    all_df = pd.read_sql("SELECT * FROM requests", conn)
    st.download_button("Download all requests CSV",
                       all_df.to_csv(index=False).encode(),
                       "all_requests.csv")

### --- MAIN APP FLOW --- ###

if not st.session_state.logged_in:
    login()
else:
    st.sidebar.write(f"👤 {st.session_state.user_type.upper()}: {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        logout()
    if st.session_state.user_type == "user":
        page = st.sidebar.radio("Go to", ["Upload", "My Requests"])
        if page == "Upload":
            user_upload_page()
        else:
            user_history_page()
    else:
        org_dashboard_page()
