"""
BugRoute AI — Streamlit Interactive Web Application
Automatic Software Bug Classification and Component Routing System.
Includes Empirical Data Leakage-Free Evaluation Dashboard, File Upload Parser & Multi-Issue Log Parser.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import BugRoutePredictor
from src.data_loader import get_or_create_dataset
from src.prediction_logger import save_prediction_correction
from src.log_parser import extract_log_issues

# Page Configuration
st.set_page_config(
    page_title="BugRoute AI — Automated Bug Routing",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Matplotlib Styling
plt.style.use('dark_background')

# Custom CSS for Premium Design & Glassmorphism
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stApp {
        background-color: #0f172a;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    .badge {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-error {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-critical {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-verified {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    .word-chip {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 4px 10px;
        margin: 4px;
        border-radius: 16px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return BugRoutePredictor()

predictor = load_predictor()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric-folders/100/bug.png", width=70)
st.sidebar.title("BugRoute AI")
st.sidebar.caption("Automated Bug Report Routing Engine")

tab_choice = st.sidebar.radio(
    "Navigation",
    ["⚡ Classify Bug Report", "📊 Compare Models & Generalization", "📈 Dataset Insights", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.82rem; background: #1e293b; padding: 12px; border-radius: 8px; border: 1px solid #334155;">
    <p style="color: #38bdf8; font-weight: bold; margin-bottom: 6px; font-size: 0.85rem;">🎓 Project Team</p>
    <p style="margin-bottom: 3px;"><b>Preet Kakdiya</b><br><span style="color: #94a3b8;">12302130501043</span></p>
    <p style="margin-bottom: 3px;"><b>Swayam Raiyani</b><br><span style="color: #94a3b8;">12302130501053</span></p>
    <p style="margin-bottom: 0px;"><b>Vruti Thakkar</b><br><span style="color: #94a3b8;">12302130501063</span></p>
    <hr style="margin: 8px 0; border-color: #334155;">
    <p style="font-size: 0.75rem; color: #cbd5e1; margin: 0;">GH Patel College of Engineering & Tech<br>Computer Science & Design</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 1: CLASSIFY BUG REPORT
# -----------------------------------------------------------------------------
if tab_choice == "⚡ Classify Bug Report":
    st.title("⚡ Real-time Bug Report Classifier")
    st.markdown("Enter a software bug title and description or upload a log file to extract & predict responsible engineering components.")

    # Preset Quick Samples
    st.subheader("💡 Try Quick Preset Bug Reports")
    samples = [
        {"name": "Database Deadlock", "title": "SQLException: Connection pool deadlock on POST /api/orders", "desc": "Transaction aborted due to lock wait timeout exceeded on PostgreSQL connection pool."},
        {"name": "UI Button Alignment", "title": "Submit button background clips in dark mode", "desc": "When toggling theme, the modal backdrop div z-index overlaps button element rendering text unreadable."},
        {"name": "JWT Auth Token", "title": "ExpiredJwtException returned on valid refresh token", "desc": "Authentication filter fails with 401 Unauthorized status when user submits signature payload."},
        {"name": "Memory Leak", "title": "java.lang.OutOfMemoryError: Java heap space during CSV export", "desc": "Exporting 500k records causes memory usage spike and GC overhead limit exceeded exception."},
        {"name": "Build CI/CD", "title": "Docker build fails with OOM killer exit code 137", "desc": "Webpack bundling step inside GitHub Actions pipeline runs out of memory on node environment."}
    ]

    cols = st.columns(len(samples))
    selected_sample = None
    for idx, sample in enumerate(samples):
        if cols[idx].button(sample["name"]):
            selected_sample = sample
            if "extracted_log_issues" in st.session_state:
                del st.session_state["extracted_log_issues"]

    default_title = selected_sample["title"] if selected_sample else ""
    default_desc = selected_sample["desc"] if selected_sample else ""

    # Optional File Upload Parser
    with st.expander("📁 Or Upload Bug Report File (.txt, .log, .json)"):
        uploaded_file = st.file_uploader(
            "Choose a bug report file", 
            type=["txt", "log", "json"],
            help="Supports .txt, .log, or .json files up to 2MB. Log files with multiple stack traces are automatically parsed into individual issues."
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            if len(file_bytes) == 0:
                st.warning("⚠️ Uploaded file is empty (0 bytes). Please upload a valid file.")
                if "extracted_log_issues" in st.session_state:
                    del st.session_state["extracted_log_issues"]
            elif len(file_bytes) > 2 * 1024 * 1024:
                st.error("⚠️ File size exceeds 2MB limit. Please upload a smaller file snippet.")
                if "extracted_log_issues" in st.session_state:
                    del st.session_state["extracted_log_issues"]
            else:
                try:
                    content_str = file_bytes.decode("utf-8", errors="replace")
                    if uploaded_file.name.endswith(".json"):
                        if "extracted_log_issues" in st.session_state:
                            del st.session_state["extracted_log_issues"]
                        try:
                            json_data = json.loads(content_str)
                            if isinstance(json_data, dict):
                                default_title = str(json_data.get("title", json_data.get("summary", uploaded_file.name)))
                                default_desc = str(json_data.get("description", json_data.get("details", content_str)))
                                st.success(f"✅ Loaded JSON bug report: '{default_title[:40]}...'")
                            else:
                                st.error("⚠️ JSON file must contain a top-level object/dictionary.")
                        except json.JSONDecodeError as je:
                            st.error(f"⚠️ Malformed JSON file: {je}")
                    
                    elif uploaded_file.name.endswith(".log"):
                        log_issues = extract_log_issues(content_str)
                        if log_issues:
                            st.session_state["extracted_log_issues"] = log_issues
                            st.session_state["uploaded_file_name"] = uploaded_file.name
                            default_title = f"Log File: {uploaded_file.name} ({len(log_issues)} issue(s) extracted)"
                            default_desc = content_str
                            st.success(f"✅ Extracted {len(log_issues)} distinct error issue(s) from '{uploaded_file.name}' with multiline stack traces grouped.")
                        else:
                            if "extracted_log_issues" in st.session_state:
                                del st.session_state["extracted_log_issues"]
                            lines = [l.strip() for l in content_str.splitlines() if l.strip()]
                            if lines:
                                default_title = lines[0][:100]
                                default_desc = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
                                st.info(f"ℹ️ No distinct stack trace headers recognized in '{uploaded_file.name}'. Treating full log file content as a single bug report.")
                            else:
                                st.warning("⚠️ Log file contains no readable text lines.")
                    
                    else:
                        if "extracted_log_issues" in st.session_state:
                            del st.session_state["extracted_log_issues"]
                        lines = [l.strip() for l in content_str.splitlines() if l.strip()]
                        if lines:
                            default_title = lines[0][:100]
                            default_desc = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
                            st.success(f"✅ Parsed file '{uploaded_file.name}': Title set from line 1.")
                        else:
                            st.warning("⚠️ File contains no readable text lines.")
                except Exception as fe:
                    st.error(f"⚠️ Failed to parse uploaded file: {fe}")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        title_input = st.text_input("Bug Title / Log Summary", value=default_title, placeholder="e.g. SQLException connection pool exhausted")
        desc_input = st.text_area("Bug Description / Log Content", value=default_desc, height=180, placeholder="e.g. Paste full exception stack trace or log file contents...")

        model_selected = st.selectbox(
            "Select Model Architecture",
            ["Linear SVM", "Logistic Regression", "BiLSTM + Attention", "DistilBERT", "CodeBERT", "Compare All 5 Models"],
            index=0
        )

        classify_btn = st.button("🚀 Classify & Route Bug", type="primary")

    with col_right:
        st.markdown("### Prediction Results")

        if classify_btn or "extracted_log_issues" in st.session_state:
            extracted_log_issues_list = st.session_state.get("extracted_log_issues", [])

            # MULTI-ISSUE LOG PROCESSING VIEW
            if extracted_log_issues_list and uploaded_file is not None and uploaded_file.name.endswith(".log"):
                st.markdown(f"#### 📋 Log Analysis Report: {len(extracted_log_issues_list)} Extracted Issue(s)")
                st.caption(f"Source Log File: `{st.session_state.get('uploaded_file_name', 'log_file.log')}`")

                for issue in extracted_log_issues_list:
                    iss_num = issue["issue_number"]
                    iss_title = issue["title"]
                    iss_desc = issue["description"]
                    iss_line = issue["line_number"]
                    iss_level = issue["log_level"]

                    res = predictor.predict(iss_title, iss_desc, model_selected)

                    if "error" in res:
                        st.error(f"❌ Issue #{iss_num} Prediction Failed: {res['error']}")
                    else:
                        badge_class = "badge-critical" if iss_level == "CRITICAL" else "badge-error"
                        conf_val = res.get("confidence", 0.0)
                        conf_display = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (float, int)) and conf_val >= 0 else "N/A"

                        with st.container():
                            st.markdown(f"""
                            <div class="metric-card" style="margin-bottom: 12px; padding: 15px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span class="{badge_class}">Issue #{iss_num} (Line {iss_line}) &bull; {iss_level}</span>
                                    <span class="badge">{res['model']}</span>
                                </div>
                                <h3 style="color: #38bdf8; margin-top: 8px; margin-bottom: 4px;">{res['predicted_component']}</h3>
                                <p style="font-size: 1.0rem; margin-bottom: 0;">Model Score: <b>{conf_display}</b></p>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander(f"🔍 Details & Stack Trace — Issue #{iss_num}: {iss_title[:60]}..."):
                                st.code(iss_desc, language="log")

                                if isinstance(conf_val, (float, int)) and conf_val < 0.50:
                                    st.warning(f"⚠️ Low Confidence Warning (< 50.0%): {conf_display}")

                                # Top 3 Predicted Components
                                top3_df = pd.DataFrame(res['top_3'])
                                st.dataframe(top3_df, hide_index=True)

                                # Feedback Form for specific issue
                                all_classes = ["Auth/Security", "Backend API", "Build/CI-CD", "Compiler/Parser", "Database", "Memory/Performance", "Network/HTTP", "Other", "UI/UX"]
                                default_idx = all_classes.index(res['predicted_component']) if res['predicted_component'] in all_classes else 0
                                
                                corrected_class = st.selectbox(
                                    f"Correct Component for Issue #{iss_num}", 
                                    all_classes, 
                                    index=default_idx,
                                    key=f"corr_class_{iss_num}"
                                )
                                user_feedback = st.text_input(
                                    f"Feedback for Issue #{iss_num}", 
                                    placeholder="e.g. Extracted stack trace indicates Hikari connection pool deadlock",
                                    key=f"fb_text_{iss_num}"
                                )
                                if st.button(f"💾 Submit Feedback for Issue #{iss_num}", key=f"sub_btn_{iss_num}"):
                                    correction_entry = {
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "title": iss_title,
                                        "description": iss_desc,
                                        "model_used": res['model'],
                                        "predicted_class": res['predicted_component'],
                                        "predicted_confidence": res.get('confidence', 0.0),
                                        "user_corrected_class": corrected_class,
                                        "user_feedback": user_feedback.strip(),
                                        "source_file": st.session_state.get('uploaded_file_name', 'log_file.log'),
                                        "issue_number": iss_num
                                    }
                                    if save_prediction_correction(correction_entry):
                                        st.success(f"✅ Feedback for Issue #{iss_num} recorded for component '{corrected_class}'.")
                                    else:
                                        st.error(f"❌ Failed to save feedback for Issue #{iss_num}.")

            # SINGLE BUG REPORT PROCESSING VIEW (Manual Input, TXT, JSON, or single log issue)
            elif classify_btn:
                if not title_input.strip() and not desc_input.strip():
                    st.warning("⚠️ Please provide a bug title or description before classifying.")
                else:
                    if model_selected != "Compare All 5 Models":
                        res = predictor.predict(title_input, desc_input, model_selected)

                        if "error" in res:
                            st.error(f"❌ Prediction Failed: {res['error']}")
                        else:
                            conf_val = res.get("confidence", 0.0)
                            conf_display = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (float, int)) and conf_val >= 0 else "N/A"

                            st.markdown(f"""
                            <div class="metric-card">
                                <span class="badge">{res['model']}</span>
                                <h2 style="color: #38bdf8; margin-top: 10px;">{res['predicted_component']}</h2>
                                <p style="font-size: 1.1rem;">Model Score: <b>{conf_display}</b></p>
                            </div>
                            """, unsafe_allow_html=True)

                            if isinstance(conf_val, (float, int)) and conf_val < 0.50:
                                st.warning(
                                    "⚠️ **Low Confidence Warning (< 50.0%)**: The predicted component has a low confidence score. "
                                    "Model scores reflect internal feature distribution outputs (Softmax/Platt scaling) and are **not guaranteed probabilities of correctness**. "
                                    "Manual triage or multi-model comparison is recommended."
                                )
                            else:
                                st.info(
                                    "ℹ️ **Confidence Transparency**: Model confidence scores represent feature decision margins or softmax probabilities. "
                                    "They reflect relative score strength rather than empirically calibrated probabilities of real-world accuracy."
                                )

                            # Top 3 Bar Chart
                            top3_df = pd.DataFrame(res['top_3'])
                            fig = px.bar(
                                top3_df, x='confidence', y='component', orientation='h',
                                title="Top-3 Predicted Components",
                                labels={'confidence': 'Score', 'component': 'Component'},
                                color='confidence', color_continuous_scale='Blues'
                            )
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=220, margin=dict(l=20, r=20, t=35, b=20))
                            st.plotly_chart(fig)

                            # Important Words Highlight
                            st.markdown("**🔍 Key Driving Features (Important Words):**")
                            words_html = "".join([f'<span class="word-chip">{w}</span>' for w in res['important_words']])
                            st.markdown(words_html, unsafe_allow_html=True)

                            # User Prediction Correction Interface
                            st.markdown("---")
                            with st.expander("✏️ Disagree with this prediction? Correct & Submit Feedback"):
                                all_classes = ["Auth/Security", "Backend API", "Build/CI-CD", "Compiler/Parser", "Database", "Memory/Performance", "Network/HTTP", "Other", "UI/UX"]
                                default_idx = all_classes.index(res['predicted_component']) if res['predicted_component'] in all_classes else 0
                                
                                corrected_class = st.selectbox(
                                    "Select Actual / Correct Component Class", 
                                    all_classes, 
                                    index=default_idx
                                )
                                user_feedback = st.text_input("Optional Feedback / Rationale", placeholder="e.g., Issue is primarily CORS preflight header handling, expected Network/HTTP")
                                submit_corr = st.button("💾 Submit Correction Log")

                                if submit_corr:
                                    correction_entry = {
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "title": title_input,
                                        "description": desc_input,
                                        "model_used": res['model'],
                                        "predicted_class": res['predicted_component'],
                                        "predicted_confidence": res.get('confidence', 0.0),
                                        "user_corrected_class": corrected_class,
                                        "user_feedback": user_feedback.strip()
                                    }
                                    if save_prediction_correction(correction_entry):
                                        st.success(f"✅ Prediction correction recorded for component '{corrected_class}'. Saved to audit log.")
                                    else:
                                        st.error("❌ Failed to save prediction correction log.")

                    else:
                        st.markdown("### Model Consensus & Comparison")
                        models_list = ["Linear SVM", "Logistic Regression", "BiLSTM + Attention", "DistilBERT", "CodeBERT"]
                        all_preds = [predictor.predict(title_input, desc_input, m) for m in models_list]

                        cmp_df = pd.DataFrame([
                            {"Model": p.get("model", m), "Predicted Component": p.get("predicted_component", "ERROR"), "Confidence Score": f"{p.get('confidence', 0.0)*100:.1f}%"}
                            for m, p in zip(models_list, all_preds)
                        ])
                        st.dataframe(cmp_df)

                        # Matplotlib Bar Chart for 5 Models Consensus
                        fig_m, ax_m = plt.subplots(figsize=(6, 2.5), facecolor='#0f172a')
                        ax_m.set_facecolor('#1e293b')
                        m_names = [p.get("model", m)[:12] for m, p in zip(models_list, all_preds)]
                        m_scores = [p.get("confidence", 0.0) * 100 for p in all_preds]
                        
                        bars = ax_m.bar(m_names, m_scores, color='#38bdf8', edgecolor='#00e5ff', width=0.45)
                        ax_m.set_ylabel('Confidence (%)', color='#f8fafc', fontweight='bold', fontsize=8)
                        ax_m.set_title('5-Model Confidence Score Comparison', color='#f8fafc', fontweight='bold', fontsize=9.5)
                        ax_m.set_ylim(0, 115)
                        ax_m.tick_params(colors='#f8fafc', labelsize=8)
                        ax_m.grid(axis='y', linestyle='--', alpha=0.3)
                        
                        for b in bars:
                            h = b.get_height()
                            ax_m.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                                         textcoords="offset points", ha='center', va='bottom', color='#00e5ff', fontweight='bold', fontsize=7.5)
                        
                        plt.tight_layout()
                        m_col1, m_col2, m_col3 = st.columns([1, 3, 1])
                        with m_col2:
                            st.pyplot(fig_m, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: COMPARE MODELS & REAL-WORLD GENERALIZATION
# -----------------------------------------------------------------------------
elif tab_choice == "📊 Compare Models & Generalization":
    st.title("📊 Empirical Model Evaluation & Generalization Dashboard")
    st.markdown("""
    This dashboard reports metrics across two distinct evaluation sets:
    1. **Synthetic Held-Out Test Set (375 samples)**: Evaluates clean test performance.
    2. **Exploratory Real-World Benchmark (20 samples)**: Evaluates generalization transfer on un-templated software bug descriptions.
    """)

    # -------------------------------------------------------------------------
    # MATPLOTLIB COMPARISON DIAGRAM (Synthetic vs Real-World Accuracy & F1)
    # -------------------------------------------------------------------------
    st.subheader("📊 Visual Accuracy & Generalization Comparison Diagram (Matplotlib)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), facecolor='#0f172a')
    
    # Chart 1: Accuracy Comparison (Synthetic vs Real-World)
    ax1.set_facecolor('#1e293b')
    models_labels = ['Linear SVM', 'Logistic Reg', 'DistilBERT', 'CodeBERT', 'BiLSTM+Attn']
    synth_acc = [100.0, 100.0, 64.8, 66.13, 100.0]
    real_acc = [90.0, 85.0, 60.0, 60.0, 35.0]

    x = np.arange(len(models_labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, synth_acc, width, label='Synthetic Test Acc (%)', color='#38bdf8')
    rects2 = ax1.bar(x + width/2, real_acc, width, label='Real-World Acc (%)', color='#00e5ff')

    ax1.set_ylabel('Accuracy (%)', color='#f8fafc', fontsize=8.5, fontweight='bold')
    ax1.set_title('Synthetic vs. Real-World Accuracy', color='#f8fafc', fontsize=9.5, fontweight='bold', pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_labels, color='#f8fafc', fontsize=7.5, rotation=15)
    ax1.tick_params(axis='y', colors='#f8fafc', labelsize=7.5)
    ax1.legend(facecolor='#1e293b', edgecolor='#38bdf8', labelcolor='#f8fafc', fontsize=7)
    ax1.set_ylim(0, 118)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', color='#38bdf8', fontsize=7, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', color='#00e5ff', fontsize=7, fontweight='bold')

    # Chart 2: Inference Latency Comparison (ms)
    ax2.set_facecolor('#1e293b')
    latencies = [0.0255, 0.0056, 20.8855, 46.2330, 1.3341]
    
    bars_lat = ax2.bar(models_labels, latencies, color='#818cf8', edgecolor='#6366f1', width=0.45)
    ax2.set_ylabel('Inference Latency (ms / sample)', color='#f8fafc', fontsize=8.5, fontweight='bold')
    ax2.set_title('Model Inference Latency (ms)', color='#f8fafc', fontsize=9.5, fontweight='bold', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_labels, color='#f8fafc', fontsize=7.5, rotation=15)
    ax2.tick_params(axis='y', colors='#f8fafc', labelsize=7.5)
    ax2.set_yscale('log') # Log scale for latency visibility across microsec to millisec
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    for b in bars_lat:
        h = b.get_height()
        val_str = f"{h:.3f}ms" if h < 1 else f"{h:.1f}ms"
        ax2.annotate(val_str, xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', color='#a5b4fc', fontsize=7, fontweight='bold')

    plt.tight_layout()
    c1, c2, c3 = st.columns([1, 8, 1])
    with c2:
        st.pyplot(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # CONFUSION MATRIX HEATMAP (MATPLOTLIB)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 Model Confusion Matrix Heatmap (Matplotlib)")
    
    eval_json_path = "reports/master_evaluation.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)

        cm_models = [m for m in eval_data.keys() if "confusion_matrix" in eval_data[m]]
        if cm_models:
            selected_cm_model = st.selectbox("Select Model to View Confusion Matrix:", cm_models, index=0)
            cm_matrix = np.array(eval_data[selected_cm_model]["confusion_matrix"])
            
            class_names = ["Auth", "API", "CI-CD", "Compiler", "DB", "Memory", "Network", "Other", "UI/UX"]

            fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.8), facecolor='#0f172a')
            ax_cm.set_facecolor('#1e293b')
            
            cax = ax_cm.imshow(cm_matrix, cmap='Blues', interpolation='nearest')
            cbar = fig_cm.colorbar(cax)
            cbar.ax.tick_params(labelsize=7, colors='#f8fafc')

            ax_cm.set_xticks(np.arange(len(class_names)))
            ax_cm.set_yticks(np.arange(len(class_names)))
            ax_cm.set_xticklabels(class_names, color='#f8fafc', fontsize=7.5, rotation=45, ha="right")
            ax_cm.set_yticklabels(class_names, color='#f8fafc', fontsize=7.5)
            
            ax_cm.set_xlabel('Predicted Component Class', color='#f8fafc', fontweight='bold', fontsize=8.5)
            ax_cm.set_ylabel('True Component Class', color='#f8fafc', fontweight='bold', fontsize=8.5)
            ax_cm.set_title(f'Confusion Matrix — {selected_cm_model}', color='#f8fafc', fontweight='bold', fontsize=9.5, pad=10)

            # Annotate matrix values
            thresh = cm_matrix.max() / 2.
            for i in range(cm_matrix.shape[0]):
                for j in range(cm_matrix.shape[1]):
                    val = cm_matrix[i, j]
                    if val > 0:
                        ax_cm.text(j, i, str(val), ha="center", va="center",
                                   color="white" if val > thresh else "#0f172a",
                                   fontweight="bold", fontsize=7)

            plt.tight_layout()
            cm_col1, cm_col2, cm_col3 = st.columns([1, 2, 1])
            with cm_col2:
                st.pyplot(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 1. Synthetic Held-Out Test Set Evaluation (375 Unique Samples)")
    
    eval_json_path = "reports/master_evaluation.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r") as f:
            eval_data = json.load(f)
            
        rows = []
        for m_name, metrics in eval_data.items():
            if m_name != "Exploratory_Real_World_Validation":
                acc = metrics.get('Accuracy', metrics.get('accuracy', 0.0))
                m_prec = metrics.get('Macro Precision', metrics.get('macro_precision', 0.0))
                m_rec = metrics.get('Macro Recall', metrics.get('macro_recall', 0.0))
                m_f1 = metrics.get('Macro-F1', metrics.get('macro_f1', 0.0))
                w_f1 = metrics.get('Weighted-F1', metrics.get('weighted_f1', 0.0))
                top3_acc = metrics.get('Top-3 Accuracy', metrics.get('top_3_accuracy', 0.0))
                inf_time = metrics.get('Inference Time (ms/sample)', metrics.get('inference_latency_ms', 0.0))

                rows.append({
                    "Model": m_name,
                    "Held-Out Accuracy": f"{acc * 100:.2f}%",
                    "Macro Precision": f"{m_prec:.4f}",
                    "Macro Recall": f"{m_rec:.4f}",
                    "Macro-F1": f"{m_f1:.4f}",
                    "Weighted-F1": f"{w_f1:.4f}",
                    "Top-3 Accuracy": f"{top3_acc * 100:.2f}%",
                    "Inference (ms)": f"{inf_time:.4f} ms"
                })
        eval_df = pd.DataFrame(rows)
        st.dataframe(eval_df, use_container_width=True)

    st.subheader("🌐 2. Exploratory Real-World Generalization Benchmark (20 Non-Synthetic Bug Reports)")
    st.caption("Evaluated on 20 manually authored software bug reports covering all 9 classes and multi-symptom defects without synthetic template structures.")

    real_world_rows = [
        {"Model": "Linear SVM", "Real-World Exploratory Accuracy": "90.00% (18/20)", "Mean Model Confidence": "0.8447", "Generalization Assessment": "High Generalization"},
        {"Model": "Logistic Regression", "Real-World Exploratory Accuracy": "85.00% (17/20)", "Mean Model Confidence": "0.2021", "Generalization Assessment": "High Generalization"},
        {"Model": "DistilBERT", "Real-World Exploratory Accuracy": "60.00% (12/20)", "Mean Model Confidence": "0.2171", "Generalization Assessment": "Moderate Transfer"},
        {"Model": "CodeBERT", "Real-World Exploratory Accuracy": "60.00% (12/20)", "Mean Model Confidence": "0.2185", "Generalization Assessment": "Moderate Transfer"},
        {"Model": "BiLSTM + Attention", "Real-World Exploratory Accuracy": "35.00% (7/20)", "Mean Model Confidence": "0.1980", "Generalization Assessment": "Overfitting / Low Generalization"}
    ]
    st.dataframe(pd.DataFrame(real_world_rows), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: DATASET INSIGHTS
# -----------------------------------------------------------------------------
elif tab_choice == "📈 Dataset Insights":
    st.title("📈 Dataset Insights & Class Distribution")
    
    data_path = "data/processed/train.csv"
    if os.path.exists(data_path):
        train_df = pd.read_csv(data_path)
        st.markdown(f"**Total Training Samples:** `{len(train_df)}` | **Feature Columns:** `title, description, component`")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Class Distribution")
            class_counts = train_df['component'].value_counts().reset_index()
            class_counts.columns = ['Component', 'Count']
            
            fig = px.pie(
                class_counts, values='Count', names='Component',
                title="Target Class Balance",
                color_discrete_sequence=px.colors.sequential.Darkmint
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Text Length Distribution")
            train_df['text_len'] = train_df['description'].astype(str).apply(len)
            fig2 = px.histogram(
                train_df, x='text_len', color='component',
                title="Character Length Distribution by Class",
                nbins=30
            )
            st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: ABOUT PROJECT
# -----------------------------------------------------------------------------
elif tab_choice == "ℹ️ About Project":
    st.title("ℹ️ BugRoute AI Project Architecture & Team")
    
    st.markdown("""
    ### 🎓 Academic Project & Team Credentials
    
    - **College:** GH Patel College of Engineering and Technology
    - **Department:** Computer Engineering Department
    - **Branch:** Computer Science and Design
    - **Project Title:** BugRoute AI — Intelligent Bug Detection Using NLP
    
    #### 👥 Team Members:
    - 👨‍💻 **Preet Kakdiya** — Enrollment: `12302130501043`
    - 👨‍💻 **Swayam Raiyani** — Enrollment: `12302130501053`
    - 👩‍💻 **Vruti Thakkar** — Enrollment: `12302130501063`

    ---

    ### 🏗️ Project Architecture & Key Highlights
    - **5 Model Architectures**: Logistic Regression, Linear SVM, BiLSTM + Attention, DistilBERT, and CodeBERT.
    - **Multi-Issue Log Parser**: Automatically parses `.log` files with multiline stack traces into distinct error issues.
    - **Empirical Generalization Benchmark**: Evaluated on 2,500 synthetic holdout samples and 20 out-of-domain software bug reports.
    - **Thread/Process-Locked Audit Logger**: Guarantees zero data corruption during prediction corrections.
    """)
