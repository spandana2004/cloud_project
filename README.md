# ♻️ GreenLoop – Smart Waste Reporting System

GreenLoop is an AI-powered waste management web application designed to empower citizens and organizations to collaborate for cleaner cities.
It allows **citizens to report dumpster waste** by uploading images, while **organizations (like NGOs or BBMP)** can monitor and accept these reports for timely collection.

Built with **Streamlit**, **YOLO object detection**, and **SQLite3**, this system demonstrates how machine learning can drive community-driven sustainability initiatives.

---

## 🚀 Features

### 👥 User Features

* 🖼️ Upload images of dumpsters or waste spots.
* 📍 Enter location details for better tracking.
* 🤖 Automatic detection of waste categories using YOLOv8 model.
* 💾 Save reports and download CSV summaries.
* 🔔 Notify NGOs or BBMP for quick response.
* 🧾 View request history with status tracking.

### 🏢 Organization Features

* 📋 Dashboard to view pending waste collection requests.
* 🖼️ Image preview and itemized waste count.
* ✅ Accept and manage cleanup requests.
* 📤 Download all request data as CSV.
* 📨 Notify users upon acceptance.

---

## 🧠 Tech Stack

| Component         | Technology                                    |
| ----------------- | --------------------------------------------- |
| **Frontend**      | Streamlit                                     |
| **Backend**       | Python, SQLite3                               |
| **AI Model**      | YOLOv8 (Ultralytics)                          |
| **Cloud Storage** | Google Drive (for model download via `gdown`) |
| **Data Handling** | Pandas, NumPy                                 |
| **Visualization** | PIL, Streamlit UI                             |

---

## 📂 Project Structure

```
GreenLoop/
│
├── uploads/                # Folder to store uploaded images
├── best.pt                 # YOLO model weights (auto-downloaded)
├── data.db                 # SQLite database
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/GreenLoop.git
cd GreenLoop
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧩 Required Dependencies

Add these to your **`requirements.txt`**:

```
streamlit
pillow
numpy
pandas
ultralytics
gdown
sqlite3-binary
```

---

## 🔐 Login Details

### Common User

* Can log in with any **name and email**.

### Organization Users

Use these pre-configured credentials:

| Organization | Email              | Password      |
| ------------ | ------------------ | ------------- |
| NGO          | `ngo@example.org`  | `password123` |
| BBMP         | `bbmp@example.gov` | `bbmp_pass`   |

---

## 🧠 Model Information

* Model used: **YOLOv8 (Ultralytics)**
* Downloaded automatically from Google Drive using:

  ```
  https://drive.google.com/uc?id=1Y_uW_GrpJthpJwHcW_0nk8eszy-a_lBN
  ```
* Model is saved as `best.pt` during first run.

---

## 🗃️ Database Schema

| Column        | Type    | Description                    |
| ------------- | ------- | ------------------------------ |
| id            | INTEGER | Primary key                    |
| user_email    | TEXT    | Email of user                  |
| user_name     | TEXT    | Name of user                   |
| location      | TEXT    | Reported location              |
| image_path    | TEXT    | Stored image file path         |
| counts_json   | TEXT    | JSON of detected object counts |
| timestamp     | TEXT    | Date/time of upload            |
| accepted      | INTEGER | 0 = Pending, 1 = Accepted      |
| accepted_by   | TEXT    | Organization email             |
| accepted_time | TEXT    | Time of acceptance             |

---

## 🧾 Example Workflow

### 🧍 User:

1. Logs in using name and email.
2. Uploads an image of a dumpster and enters its location.
3. YOLO model detects waste objects (e.g., “plastic bottle”, “can”).
4. User downloads a report or sends a notification to organizations.

### 🏢 Organization:

1. Logs in using organization credentials.
2. Views pending cleanup requests.
3. Accepts a request → notifies the reporting user.
4. Downloads complete request data as CSV for recordkeeping.

---

## 💡 Future Enhancements

* 🔗 Integration with Google Maps API for precise location tagging.
* 📧 Email notifications using SendGrid or SMTP.
* 🧭 AI-based route optimization for waste collection.
* 📱 Mobile app version (Android/iOS).
* 🗺️ Dashboard analytics for waste trend visualization.

---
