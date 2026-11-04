import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import base64
import requests

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# BACKGROUND IMAGE
# =========================
def set_bg_image():
    image_path = r"C:\Users\dell\OneDrive\ドキュメント\4112932.jpg"
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Dark overlay */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.55);
    z-index: 0;
}}

/* Keep content visible */
.main {{
    position: relative;
    z-index: 1;
}}

/* Login form styling */
[data-testid="stForm"] {{
    background: rgba(0,0,0,0.75);
    padding: 30px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}}
</style>
""", unsafe_allow_html=True)
    except:
        pass

set_bg_image()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9f43 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .doctor-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #4fc3f7;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }
    .section-box {
        background: rgba(0,0,0,0.6);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .chat-user {
        background: linear-gradient(135deg, #1e3a5f, #2a5298);
        color: #ffffff !important;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 80px;
        font-size: 0.97rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .chat-bot {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #e0e0e0 !important;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 80px 8px 0;
        font-size: 0.97rem;
        line-height: 1.6;
        border-left: 4px solid #4fc3f7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .chat-label-user {
        color: #90caf9 !important;
        font-weight: bold;
        font-size: 0.85rem;
        text-align: right;
        margin-right: 4px;
    }
    .chat-label-bot {
        color: #4fc3f7 !important;
        font-weight: bold;
        font-size: 0.85rem;
        margin-left: 4px;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        border: 2px solid #4fc3f7 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    .stTextInput>div>div>input::placeholder {
        color: #aaaaaa !important;
    }
    .action-buttons-gap {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'appointments' not in st.session_state:
    st.session_state.appointments = []

# =========================
# CONSTANTS
# =========================
FEATURE_GROUPS = {
    "Basic Information": [
        ('Age', 'Age (years)', 15, 60, 28),
        ('BMI', 'BMI', 15.0, 45.0, 32.0),
        ('Waist_Circumference', 'Waist Circumference (cm)', 50, 150, 98),
    ],
    "Blood & Metabolic": [
        ('Insulin_Levels', 'Insulin Levels (μU/mL)', 2.0, 30.0, 20.0),
        ('Blood_Glucose', 'Blood Glucose Level', 70.0, 250.0, 148.0),
        ('BP_Systolic', 'Systolic BP', 90, 180, 138),
        ('BP_Diastolic', 'Diastolic BP', 60, 120, 90),
    ],
    "Lifestyle & Family": [
        ('Family_History', 'Family History (0=No, 1=Yes)', 0, 1, 1),
        ('Physical_Activity', 'Physical Activity Level (1-5)', 1, 5, 1),
        ('Fast_Food', 'Fast Food Consumption (0=No,1=Yes)', 0, 1, 1),
    ],
    "PCOS Symptoms": [
        ('Weight_Gain', 'Weight Gain (0=No,1=Yes)', 0, 1, 1),
        ('Hair_Growth', 'Hair Growth (0=No,1=Yes)', 0, 1, 1),
        ('Pimples', 'Pimples (0=No,1=Yes)', 0, 1, 1),
    ],
    "Reproductive Health": [
        ('Follicle_No_L', 'Follicle Count Left', 0, 20, 14),
        ('Follicle_No_R', 'Follicle Count Right', 0, 20, 14),
        ('Endometrium', 'Endometrium Thickness', 2.0, 20.0, 13.0),
    ],
}

HIGH_RISK_DEFAULTS = {
    'Age': 28, 'BMI': 32.0, 'Waist_Circumference': 98,
    'Insulin_Levels': 20.0, 'Blood_Glucose': 148.0,
    'BP_Systolic': 138, 'BP_Diastolic': 90,
    'Family_History': 1, 'Physical_Activity': 1, 'Fast_Food': 1,
    'Weight_Gain': 1, 'Hair_Growth': 1, 'Pimples': 1,
    'Follicle_No_L': 14, 'Follicle_No_R': 14, 'Endometrium': 13.0,
}

LOW_RISK_DEFAULTS = {
    'Age': 25, 'BMI': 22.0, 'Waist_Circumference': 70,
    'Insulin_Levels': 7.0, 'Blood_Glucose': 85.0,
    'BP_Systolic': 110, 'BP_Diastolic': 70,
    'Family_History': 0, 'Physical_Activity': 5, 'Fast_Food': 0,
    'Weight_Gain': 0, 'Hair_Growth': 0, 'Pimples': 0,
    'Follicle_No_L': 4, 'Follicle_No_R': 4, 'Endometrium': 6.0,
}

# =========================
# DOCTOR DATABASE
# =========================
DOCTORS = {
    "Endocrinologist": [
        {"name": "Dr. Priya Sharma", "hospital": "Apollo Hospital, Chennai",
         "experience": "15 years", "rating": "⭐ 4.9", "fee": "₹800",
         "available": "Mon, Wed, Fri", "phone": "+91-9876543210"},
        {"name": "Dr. Meena Krishnan", "hospital": "Fortis Hospital, Chennai",
         "experience": "12 years", "rating": "⭐ 4.8", "fee": "₹700",
         "available": "Tue, Thu, Sat", "phone": "+91-9876543211"},
        {"name": "Dr. Rajesh Kumar", "hospital": "MIOT Hospital, Chennai",
         "experience": "18 years", "rating": "⭐ 4.7", "fee": "₹900",
         "available": "Mon, Tue, Wed", "phone": "+91-9876543212"},
    ],
    "Gynecologist": [
        {"name": "Dr. Anitha Rajan", "hospital": "Cloudnine Hospital, Chennai",
         "experience": "20 years", "rating": "⭐ 4.9", "fee": "₹600",
         "available": "Mon, Wed, Sat", "phone": "+91-9876543213"},
        {"name": "Dr. Lakshmi Devi", "hospital": "Vijaya Hospital, Chennai",
         "experience": "10 years", "rating": "⭐ 4.6", "fee": "₹500",
         "available": "Tue, Thu, Fri", "phone": "+91-9876543214"},
        {"name": "Dr. Kavitha Nair", "hospital": "Sri Ramachandra Hospital",
         "experience": "14 years", "rating": "⭐ 4.8", "fee": "₹650",
         "available": "Mon, Thu, Sat", "phone": "+91-9876543215"},
    ],
    "General Physician": [
        {"name": "Dr. Suresh Babu", "hospital": "Kauvery Hospital, Chennai",
         "experience": "8 years", "rating": "⭐ 4.5", "fee": "₹400",
         "available": "Mon to Sat", "phone": "+91-9876543216"},
        {"name": "Dr. Divya Menon", "hospital": "Global Hospital, Chennai",
         "experience": "6 years", "rating": "⭐ 4.4", "fee": "₹350",
         "available": "Mon to Fri", "phone": "+91-9876543217"},
        {"name": "Dr. Arun Prakash", "hospital": "Stanley Medical College",
         "experience": "11 years", "rating": "⭐ 4.6", "fee": "₹450",
         "available": "Tue, Thu, Sat", "phone": "+91-9876543218"},
    ],
    "Nutritionist": [
        {"name": "Dr. Pooja Iyer", "hospital": "Diet & Wellness Clinic, Chennai",
         "experience": "7 years", "rating": "⭐ 4.7", "fee": "₹300",
         "available": "Mon, Wed, Fri", "phone": "+91-9876543219"},
        {"name": "Dr. Saranya Tamil", "hospital": "HealthFirst Center, Chennai",
         "experience": "5 years", "rating": "⭐ 4.5", "fee": "₹250",
         "available": "Tue, Thu, Sat", "phone": "+91-9876543220"},
        {"name": "Dr. Nithya Balaji", "hospital": "Nutrition Plus, Chennai",
         "experience": "9 years", "rating": "⭐ 4.8", "fee": "₹350",
         "available": "Mon to Fri", "phone": "+91-9876543221"},
    ],
}

# =========================
# DATA GENERATION
# =========================
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    pos = {
        'Age':               np.random.randint(18, 42, n_pos),
        'BMI':               np.clip(np.random.normal(27, 6, n_pos), 16, 45),
        'Waist_Circumference': np.clip(np.random.normal(88, 12, n_pos), 60, 130),
        'Insulin_Levels':    np.clip(np.random.normal(15, 6, n_pos), 4, 30),
        'Blood_Glucose':     np.clip(np.random.normal(125, 30, n_pos), 75, 250),
        'BP_Systolic':       np.random.randint(108, 158, n_pos),
        'BP_Diastolic':      np.random.randint(68, 102, n_pos),
        'Family_History':    np.random.choice([0, 1], n_pos, p=[0.35, 0.65]),
        'Physical_Activity': np.random.choice([1, 2, 3, 4, 5], n_pos, p=[0.25, 0.30, 0.25, 0.12, 0.08]),
        'Fast_Food':         np.random.choice([0, 1], n_pos, p=[0.30, 0.70]),
        'Weight_Gain':       np.random.choice([0, 1], n_pos, p=[0.25, 0.75]),
        'Hair_Growth':       np.random.choice([0, 1], n_pos, p=[0.30, 0.70]),
        'Pimples':           np.random.choice([0, 1], n_pos, p=[0.35, 0.65]),
        'Follicle_No_L':     np.clip(np.random.normal(11, 4, n_pos), 4, 20).astype(int),
        'Follicle_No_R':     np.clip(np.random.normal(11, 4, n_pos), 4, 20).astype(int),
        'Endometrium':       np.clip(np.random.normal(10.5, 3, n_pos), 4, 20),
        'PCOS_Diagnosis':    np.ones(n_pos, dtype=int),
    }

    neg = {
        'Age':               np.random.randint(18, 45, n_neg),
        'BMI':               np.clip(np.random.normal(24, 5, n_neg), 16, 38),
        'Waist_Circumference': np.clip(np.random.normal(78, 11, n_neg), 55, 110),
        'Insulin_Levels':    np.clip(np.random.normal(10, 5, n_neg), 2, 22),
        'Blood_Glucose':     np.clip(np.random.normal(96, 22, n_neg), 70, 145),
        'BP_Systolic':       np.random.randint(100, 140, n_neg),
        'BP_Diastolic':      np.random.randint(60, 92, n_neg),
        'Family_History':    np.random.choice([0, 1], n_neg, p=[0.70, 0.30]),
        'Physical_Activity': np.random.choice([1, 2, 3, 4, 5], n_neg, p=[0.10, 0.18, 0.27, 0.28, 0.17]),
        'Fast_Food':         np.random.choice([0, 1], n_neg, p=[0.65, 0.35]),
        'Weight_Gain':       np.random.choice([0, 1], n_neg, p=[0.75, 0.25]),
        'Hair_Growth':       np.random.choice([0, 1], n_neg, p=[0.75, 0.25]),
        'Pimples':           np.random.choice([0, 1], n_neg, p=[0.70, 0.30]),
        'Follicle_No_L':     np.clip(np.random.normal(7, 3.5, n_neg), 2, 14).astype(int),
        'Follicle_No_R':     np.clip(np.random.normal(7, 3.5, n_neg), 2, 14).astype(int),
        'Endometrium':       np.clip(np.random.normal(8, 2.5, n_neg), 2, 13),
        'PCOS_Diagnosis':    np.zeros(n_neg, dtype=int),
    }

    df = pd.concat([pd.DataFrame(pos), pd.DataFrame(neg)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    noise_idx = df.sample(frac=0.03, random_state=7).index
    df.loc[noise_idx, 'PCOS_Diagnosis'] = 1 - df.loc[noise_idx, 'PCOS_Diagnosis']
    return df

# =========================
# CHATBOT FUNCTIONS
# =========================
def smart_fallback_response(query):
    q = query.lower()

    if any(w in q for w in ["diet", "food", "eat", "nutrition", "meal", "drink", "fruit", "vegetable", "carb", "sugar", "calorie"]):
        return """**Diet to Reduce Diabetes & PCOS Risk:**

**Best Foods to Eat:**
- Whole grains: oats, brown rice, quinoa, millets
- Vegetables: spinach, broccoli, bitter gourd, methi
- Fruits: berries, apple, guava, papaya (low GI)
- Proteins: eggs, fish, chicken, lentils, tofu
- Healthy fats: nuts, seeds, avocado, olive oil

**Foods to Strictly Avoid:**
- White rice, white bread, maida products
- Sugary drinks, fruit juices, soda
- Sweets, chocolates, biscuits, pastries
- Deep fried and processed foods

**Meal Tips:**
- Eat every 3-4 hours in small portions
- Never skip breakfast
- Drink 8-10 glasses of water daily
- Add cinnamon and turmeric to food

Please consult a dietitian for a personalized meal plan."""

    elif any(w in q for w in ["exercise", "workout", "gym", "walk", "physical", "activity", "yoga", "sport", "run", "fit"]):
        return """**Best Exercises to Improve Diabetes & PCOS:**

**Cardio (5 days/week):**
- Brisk walking: 30-45 mins daily
- Cycling: 20-30 mins
- Swimming: 30 mins
- Zumba or aerobics: 30 mins

**Strength Training (2-3 days/week):**
- Squats, lunges, push-ups
- Resistance band exercises
- Light dumbbell workouts

**Yoga (Daily, 20 mins):**
- Surya Namaskar: balances hormones
- Kapalbhati: improves metabolism
- Butterfly pose: good for PCOS

**Weekly Goal:** 150 minutes moderate exercise"""

    elif any(w in q for w in ["symptom", "sign", "feel", "pain", "tired", "fatigue", "period", "irregular", "hair", "acne"]):
        return """**Symptoms You Should Know About:**

**Early Diabetes Warning Signs:**
- Urinating more than usual especially at night
- Feeling very thirsty even after drinking water
- Extreme hunger despite eating
- Unexplained weight loss
- Blurry or worsening vision
- Cuts and wounds healing very slowly
- Tingling or numbness in feet or hands

**PCOS Warning Signs:**
- Irregular, missed, or very heavy periods
- Acne on face, chest, or back
- Unusual hair growth on face or body
- Hair thinning or falling from scalp
- Dark skin patches on neck or underarms
- Difficulty getting pregnant
- Mood swings, anxiety, or depression

Please consult a doctor if you notice 3 or more symptoms."""

    elif any(w in q for w in ["improve", "better", "healthy", "health", "lifestyle", "prevent", "risk", "reduce"]):
        return """**How to Improve Your Health & Reduce Risk:**

**1. Fix Your Diet:**
- Replace white rice with millets or brown rice
- Cut out sugar and sugary drinks completely
- Eat more vegetables, fiber, and protein

**2. Move Your Body:**
- Walk at least 30 minutes every day
- Add strength training 2-3 times per week
- Yoga helps balance hormones

**3. Sleep Well:**
- Sleep 7-9 hours every night
- Poor sleep raises blood sugar and cortisol

**4. Manage Stress:**
- Practice deep breathing or meditation daily
- High stress worsens both PCOS and diabetes

**5. Monitor Regularly:**
- Check fasting blood sugar every 3 months
- Get HbA1c test done annually"""

    elif any(w in q for w in ["medicine", "medication", "tablet", "pill", "drug", "treatment", "metformin", "insulin"]):
        return """**Medications Used for Diabetes & PCOS:**

**Common Diabetes Medications:**
- Metformin: Most common first medicine
- Glipizide / Glimepiride: Stimulates insulin release
- SGLT2 inhibitors (Empagliflozin): Remove sugar through urine
- GLP-1 agonists (Semaglutide): Control sugar and help weight loss
- Insulin injections: Used when tablets are not enough

**Common PCOS Medications:**
- Metformin: Improves insulin resistance
- Birth control pills: Regulate periods
- Spironolactone: Reduces male hormone effects
- Clomiphene / Letrozole: Help with ovulation
- Inositol supplements: Natural option

Never start or stop medications without doctor guidance."""

    elif any(w in q for w in ["weight", "fat", "obese", "bmi", "overweight", "lose weight", "belly"]):
        return """**Weight Management for Diabetes & PCOS:**

**Why Weight Matters:**
- Even 5-10% weight loss improves blood sugar significantly
- Losing belly fat improves insulin sensitivity
- Weight loss can restore regular periods in PCOS

**Effective Strategies:**
- Create a 300-500 calorie daily deficit
- Eat high-protein, high-fiber, low-carb meals
- Combine cardio and strength training
- HIIT burns more fat in less time
- Aim for 10,000 steps daily
- Sleep 7-9 hours (poor sleep causes weight gain)

Realistic Goal: Aim for 0.5-1 kg per week loss."""

    elif any(w in q for w in ["blood sugar", "glucose", "hba1c", "fasting", "sugar level", "normal range"]):
        return """**Understanding Blood Sugar Levels:**

**Normal Ranges:**
- Fasting (8 hrs no food): 70-100 mg/dL = Normal
- Fasting: 100-125 mg/dL = Prediabetes
- Fasting: 126+ mg/dL = Diabetes
- After 2 hrs of eating: Below 140 = Normal
- HbA1c below 5.7% = Normal
- HbA1c 5.7-6.4% = Prediabetes
- HbA1c 6.5% and above = Diabetes

**How to Lower High Blood Sugar:**
- Avoid refined carbs and sugar immediately
- Walk for 15-20 mins after every meal
- Drink plenty of water
- Take prescribed medications on time

**Emergency:** Seek care if blood sugar is above 400 mg/dL"""

    elif any(w in q for w in ["pcos", "polycystic", "ovary", "hormone", "period", "fertility", "pregnant"]):
        return """**PCOS Complete Guide:**

**What is PCOS?**
A hormonal disorder where ovaries produce excess male hormones causing irregular periods.

**Main Causes:**
- Insulin resistance (most common)
- Genetics and family history
- Chronic inflammation

**Natural Management:**
- Low glycemic index foods reduce insulin resistance
- Anti-inflammatory foods: turmeric, ginger, berries
- Regular exercise reduces insulin resistance by up to 40%
- Yoga specifically helps balance female hormones

**Helpful Supplements:**
- Inositol (Myo-inositol + D-chiro-inositol)
- Vitamin D, Omega-3, Magnesium

Please consult a gynecologist for personalized PCOS management."""

    elif any(w in q for w in ["stress", "anxiety", "depression", "mental", "mood", "sleep", "rest"]):
        return """**Managing Stress for Better Health:**

**Why Stress Worsens Your Condition:**
- Stress releases cortisol which raises blood sugar
- Cortisol causes belly fat accumulation
- Chronic stress disrupts menstrual cycle in PCOS

**Effective Stress Management:**
- Deep breathing: 5 mins morning and night
- Meditation: Even 10 mins daily reduces cortisol
- Yoga reduces cortisol better than most exercises
- Walking in nature: 20 mins improves mood

**Sleep Hygiene:**
- No screens 1 hour before bed
- Keep room cool and dark
- Avoid caffeine after 2 PM

If anxiety is severe, please speak to a mental health professional."""

    else:
        return f"""Thank you for your question about: **"{query}"**

I can provide detailed information on:
- 🩸 **Blood Sugar & Diabetes** — levels, control, medication
- 🌿 **PCOS** — symptoms, hormones, fertility, management
- 🥗 **Diet & Nutrition** — what to eat, meal planning
- 🏃 **Exercise** — best workouts, weekly plans
- ⚖️ **Weight Management** — how to lose weight safely
- 💊 **Medications** — common drugs, how they work
- 😴 **Sleep & Stress** — impact on health
- 🩺 **Symptoms** — when to worry, when to see a doctor

Try asking: "What should I eat to lower blood sugar?" or "What are the symptoms of PCOS?"

Please consult a qualified healthcare professional for personalized advice."""


def get_ai_response(user_question, chat_history=None):
    SYSTEM_PROMPT = """You are a knowledgeable and compassionate health assistant specializing in
PCOS, diabetes, women's health, nutrition, and lifestyle medicine.
- Give a DIRECT and SPECIFIC answer to exactly what the user asked
- Use bullet points or numbered steps when listing information
- Mention when to consult a doctor for serious concerns
- Be warm, supportive, and encouraging
- Never repeat the same answer for different questions"""

    messages = []
    if chat_history:
        for sender, msg in chat_history[-6:]:
            if sender == "You":
                messages.append({"role": "user", "content": msg})
            else:
                messages.append({"role": "assistant", "content": msg})
    messages.append({"role": "user", "content": user_question})

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": messages
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"]
        else:
            return smart_fallback_response(user_question)
    except Exception:
        return smart_fallback_response(user_question)

# =========================
# LOGIN PAGE  (Demo login removed)
# =========================
def login_page():
    st.markdown('<h1 class="main-header">🏥 Diabetes Prediction System</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Welcome! Please login to continue")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("🔐 Login", use_container_width=True)

            if login_btn:
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.current_page = "Dashboard"
                    st.rerun()
                else:
                    st.error("Please enter both username and password")

# =========================
# DASHBOARD PAGE
# =========================
def dashboard_page():
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"### Welcome back, {st.session_state.username}! 👋")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <h2>🧬</h2><h3>16</h3><p>Features Analyzed</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        model_status = "✅ Trained" if st.session_state.model_trained else "⚠️ Not Trained"
        st.markdown(f"""<div class="metric-card">
            <h2>🤖</h2><h3>{model_status}</h3><p>Model Status</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h2>📈</h2><h3>95%</h3><p>Accuracy Target</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        appt_count = len(st.session_state.appointments)
        st.markdown(f"""<div class="metric-card">
            <h2>📅</h2><h3>{appt_count}</h3><p>Appointments Booked</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎯 Train Model", use_container_width=True):
            st.session_state.current_page = "Train Model"
            st.rerun()
    with col2:
        if st.button("📝 Patient Input", use_container_width=True):
            st.session_state.current_page = "Patient Input"
            st.rerun()
    with col3:
        if st.button("👨‍⚕️ Find Doctors", use_container_width=True):
            st.session_state.current_page = "Doctor Recommendation"
            st.rerun()
    with col4:
        if st.button("🤖 Health Chatbot", use_container_width=True):
            st.session_state.current_page = "Chatbot"
            st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 About This System")
        st.markdown("""
        This system uses machine learning to assess PCOS and diabetes risk:
        - 🧬 Genetic markers and family history
        - 📊 Metabolic indicators (BMI, insulin, glucose)
        - 💪 Lifestyle factors (diet, exercise)
        - 🩺 Clinical symptoms and test results
        - 👨‍⚕️ Doctor recommendations based on risk
        - 📅 Online and in-person appointment booking
        """)
    with col2:
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Train the Model** — Use sample data or upload your own
        2. **Input Patient Data** — Enter all health parameters
        3. **Get Prediction** — View PCOS risk percentage
        4. **Find Doctors** — Get specialist recommendations
        5. **Book Appointment** — In-person or online consultation
        6. **Use Chatbot** — Ask any health question

        ⚠️ **For screening purposes only. Not a substitute for medical diagnosis.**
        """)

# =========================
# TRAIN MODEL PAGE
# =========================
def train_model_page():
    st.markdown('<h1 class="main-header">🎯 Train Prediction Model</h1>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📊 Use Sample Data", "📁 Upload Custom Dataset"])

    with tab1:
        st.markdown("### Generate and Train with Sample Data")
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Number of samples", 500, 5000, 3000, 100)
        with col2:
            test_size = st.slider("Test set size (%)", 10, 40, 20)

        model_type = st.selectbox("Select Model Algorithm",
                                  ["Random Forest", "Gradient Boosting", "Logistic Regression"])

        if st.button("🚀 Generate Data & Train Model", use_container_width=True):
            with st.spinner("Generating dataset..."):
                df = generate_sample_data(n_samples)
                st.success(f"✅ Generated {n_samples} samples")

            st.dataframe(df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                target_counts = df['PCOS_Diagnosis'].value_counts()
                st.bar_chart(target_counts)
            with col2:
                st.write(f"- Total samples: {len(df)}")
                st.write(f"- PCOS Positive: {target_counts.get(1, 0)}")
                st.write(f"- PCOS Negative: {target_counts.get(0, 0)}")

            with st.spinner("Training model..."):
                feature_cols = [col for col in df.columns if col != 'PCOS_Diagnosis']
                X = df[feature_cols]
                y = df['PCOS_Diagnosis']
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if model_type == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=300, max_depth=15,
                        min_samples_split=4, min_samples_leaf=2,
                        max_features='sqrt', random_state=42, n_jobs=-1)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=300, learning_rate=0.08,
                        max_depth=5, subsample=0.85, random_state=42)
                else:
                    model = LogisticRegression(max_iter=5000, C=0.8, random_state=42)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.model_trained = True
                st.session_state.feature_columns = feature_cols

            st.success("✅ Model trained successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            with col2:
                st.metric("Training Samples", len(X_train))
            with col3:
                st.metric("Test Samples", len(X_test))

            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    with tab2:
        st.markdown("### Upload Your Own Dataset")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)
            target_col = st.selectbox("Select Target Column", df.columns.tolist())

            if st.button("🚀 Train with Uploaded Data", use_container_width=True):
                with st.spinner("Training model..."):
                    from sklearn.preprocessing import LabelEncoder
                    feature_cols = [col for col in df.columns if col != target_col]
                    X = df[feature_cols].copy()
                    y = df[target_col].copy()

                    if y.dtype == object:
                        le = LabelEncoder()
                        y = pd.Series(le.fit_transform(y), name=target_col)

                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].fillna('Unknown').astype(str))

                    X = X.fillna(X.median(numeric_only=True))
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(scaler.fit_transform(X_train), y_train)
                    y_pred = model.predict(scaler.transform(X_test))
                    accuracy = accuracy_score(y_test, y_pred)

                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.model_trained = True
                    st.session_state.feature_columns = list(X.columns)

                st.success(f"✅ Model trained with accuracy: {accuracy*100:.2f}%")

# =========================
# PATIENT INPUT PAGE
# =========================
def patient_input_page():
    st.markdown('<h1 class="main-header">📝 Patient Health Data Input</h1>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first.")
        if st.button("Go to Train Model"):
            st.session_state.current_page = "Train Model"
            st.rerun()
        return

    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("🔴 Load HIGH RISK Patient", use_container_width=True):
            for k, v in HIGH_RISK_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
    with qc2:
        if st.button("🟢 Load LOW RISK Patient", use_container_width=True):
            for k, v in LOW_RISK_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
    with qc3:
        if st.button("🔄 Reset to Default", use_container_width=True):
            for group_name, features in FEATURE_GROUPS.items():
                for feature_key, label, min_val, max_val, default_val in features:
                    st.session_state[feature_key] = default_val
            st.rerun()

    def _get(key, fallback):
        return st.session_state.get(key, fallback)

    input_data = {}
    with st.form("patient_form"):
        for group_name, features in FEATURE_GROUPS.items():
            st.markdown(f"### {group_name}")
            cols = st.columns(3)
            for idx, (feature_key, label, min_val, max_val, default_val) in enumerate(features):
                with cols[idx % 3]:
                    current_val = _get(feature_key, default_val)
                    if isinstance(min_val, int):
                        input_data[feature_key] = st.number_input(
                            label, min_value=min_val, max_value=max_val,
                            value=int(current_val), key=f"form_{feature_key}")
                    else:
                        input_data[feature_key] = st.number_input(
                            label, min_value=float(min_val), max_value=float(max_val),
                            value=float(current_val), key=f"form_{feature_key}")
            st.markdown("---")

        submitted = st.form_submit_button("🔮 Analyze & Get Prediction", use_container_width=True)
        if submitted:
            feature_cols = st.session_state.feature_columns
            input_df = pd.DataFrame([input_data])
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_cols]
            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.model.predict(input_scaled)[0]
            probability = st.session_state.model.predict_proba(input_scaled)[0]
            risk_score = probability[1] * 100

            st.session_state.prediction_result = {
                'prediction': prediction,
                'risk_score': risk_score,
                'input_data': input_data
            }
            st.rerun()

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        risk = result['risk_score']
        st.markdown("---")
        st.markdown("## 🔮 Prediction Result")

        if risk >= 65:
            st.markdown(f"""<div class="risk-high">
                <h2>⚠️ HIGH RISK</h2>
                <h1>{risk:.1f}%</h1>
                <p>PCOS/Diabetes Risk Score</p>
                <p>Immediate medical consultation recommended</p>
            </div>""", unsafe_allow_html=True)
        elif risk >= 35:
            st.markdown(f"""<div class="risk-medium">
                <h2>⚡ MODERATE RISK</h2>
                <h1>{risk:.1f}%</h1>
                <p>PCOS/Diabetes Risk Score</p>
                <p>Lifestyle changes and monitoring advised</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risk-low">
                <h2>✅ LOW RISK</h2>
                <h1>{risk:.1f}%</h1>
                <p>PCOS/Diabetes Risk Score</p>
                <p>Maintain healthy habits and regular checkups</p>
            </div>""", unsafe_allow_html=True)

        # ── Gap between risk card and action buttons ──
        st.markdown("<br><br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👨‍⚕️ Find Recommended Doctors", use_container_width=True):
                st.session_state.current_page = "Doctor Recommendation"
                st.rerun()
        with col2:
            if st.button("🤖 Ask Health Chatbot", use_container_width=True):
                st.session_state.current_page = "Chatbot"
                st.rerun()

# =========================
# DOCTOR RECOMMENDATION PAGE
# =========================
def doctor_recommendation_page():
    st.markdown('<h1 class="main-header">👨‍⚕️ Doctor Recommendations</h1>', unsafe_allow_html=True)

    risk_score = 50
    if st.session_state.prediction_result:
        risk_score = st.session_state.prediction_result['risk_score']
        st.info(f"📊 Based on your risk score: **{risk_score:.1f}%**")

    specialty_filter = st.selectbox(
        "Filter by Specialty",
        ["All Specialists"] + list(DOCTORS.keys())
    )

    specialties_to_show = list(DOCTORS.keys()) if specialty_filter == "All Specialists" else [specialty_filter]

    for specialty in specialties_to_show:
        st.markdown(f"### 🏥 {specialty}")
        cols = st.columns(3)
        for i, doc in enumerate(DOCTORS[specialty]):
            with cols[i % 3]:
                st.markdown(f"""<div class="doctor-card">
                    <h4>👨‍⚕️ {doc['name']}</h4>
                    <p>🏥 {doc['hospital']}</p>
                    <p>⏳ Experience: {doc['experience']}</p>
                    <p>{doc['rating']} | 💰 {doc['fee']}</p>
                    <p>📅 {doc['available']}</p>
                    <p>📞 {doc['phone']}</p>
                </div>""", unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"📅 Book", key=f"book_{specialty}_{i}", use_container_width=True):
                        appointment = {
                            'doctor': doc['name'],
                            'hospital': doc['hospital'],
                            'specialty': specialty,
                            'type': 'In-Person',
                            'fee': doc['fee']
                        }
                        st.session_state.appointments.append(appointment)
                        st.success(f"✅ Booked with {doc['name']}!")
                with col_b:
                    if st.button(f"💻 Online", key=f"online_{specialty}_{i}", use_container_width=True):
                        appointment = {
                            'doctor': doc['name'],
                            'hospital': doc['hospital'],
                            'specialty': specialty,
                            'type': 'Online',
                            'fee': doc['fee']
                        }
                        st.session_state.appointments.append(appointment)
                        st.success(f"✅ Online consult booked with {doc['name']}!")

    if st.session_state.appointments:
        st.markdown("---")
        st.markdown("### 📅 Your Booked Appointments")
        for i, appt in enumerate(st.session_state.appointments):
            st.markdown(f"""
            **{i+1}. {appt['doctor']}** — {appt['specialty']}
            🏥 {appt['hospital']} | 💻 {appt['type']} | 💰 {appt['fee']}
            """)

# =========================
# CHATBOT PAGE
# =========================
def chatbot_page():
    st.markdown('<h1 class="main-header">🤖 Health Assistant Chatbot</h1>', unsafe_allow_html=True)

    if st.session_state.prediction_result:
        risk = st.session_state.prediction_result['risk_score']
        st.info(f"📊 Your current risk score: **{risk:.1f}%** — Ask me anything about managing your health!")

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""<div class="chat-bot">
                👋 Hello! I'm your AI Health Assistant. I can help you with:
                <br>• 🩸 Diabetes & blood sugar management
                <br>• 🌿 PCOS symptoms and treatment
                <br>• 🥗 Diet and nutrition advice
                <br>• 🏃 Exercise recommendations
                <br>• 💊 Medication information
                <br><br>What would you like to know?
            </div>""", unsafe_allow_html=True)
        else:
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f'<div class="chat-label-user">You</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-user">{message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-label-bot">🤖 Health Assistant</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-bot">{message}</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask a health question...",
                placeholder="e.g. What foods should I avoid with PCOS?",
                label_visibility="collapsed"
            )
        with col2:
            send_btn = st.form_submit_button("Send 💬", use_container_width=True)

        if send_btn and user_input.strip():
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Thinking..."):
                response = get_ai_response(user_input, st.session_state.chat_history)
            st.session_state.chat_history.append(("Bot", response))
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**💡 Quick Questions:**")
        quick_questions = [
            "What foods help reduce blood sugar?",
            "What are PCOS symptoms?",
            "Best exercises for diabetes?",
            "How to manage stress with PCOS?"
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                st.session_state.chat_history.append(("You", q))
                response = get_ai_response(q, st.session_state.chat_history)
                st.session_state.chat_history.append(("Bot", response))
                st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# =========================
# MAIN APP
# =========================
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown("---")

        pages = ["Dashboard", "Train Model", "Patient Input", "Doctor Recommendation", "Chatbot"]
        for page in pages:
            icon_map = {
                "Dashboard": "📊",
                "Train Model": "🎯",
                "Patient Input": "📝",
                "Doctor Recommendation": "👨‍⚕️",
                "Chatbot": "🤖"
            }
            if st.button(f"{icon_map[page]} {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()

        st.markdown("---")
        if st.session_state.model_trained:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model Not Trained")

        if st.session_state.prediction_result:
            risk = st.session_state.prediction_result['risk_score']
            st.metric("Last Risk Score", f"{risk:.1f}%")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_page = "Login"
            st.rerun()

    page = st.session_state.current_page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Patient Input":
        patient_input_page()
    elif page == "Doctor Recommendation":
        doctor_recommendation_page()
    elif page == "Chatbot":
        chatbot_page()


if __name__ == "__main__":
    main()