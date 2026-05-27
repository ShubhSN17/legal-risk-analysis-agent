import streamlit as st
import requests
import pandas as pd

# 1. Page Configuration (Wide Layout for Dashboard feel)
st.set_page_config(page_title="Legal Risk Agent", page_icon="⚖️", layout="wide")

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("API Connection: Online")
    st.info("Model: Llama 3.3 70B Versatile")
    st.info("Vector DB: ChromaDB (Cached)")
    st.markdown("---")
    st.markdown("**Architecture:**\n- Frontend: Streamlit\n- Backend: FastAPI\n- AI Engine: LangGraph\n- RAG: HuggingFace + Chroma")

# 3. Main Header
st.title("⚖️ Legal Contract Risk Analyzer")
st.markdown("Upload a third-party contract. The autonomous agent will chunk the document, retrieve relevant internal policies, and strictly adjudicate the text for legal risk.")
st.markdown("---")

# 4. Upload Area
uploaded_file = st.file_uploader("Drop your PDF contract here", type="pdf")

if st.button("Run AI Risk Analysis", type="primary"):
    if uploaded_file is not None:
        with st.spinner("Agent is reading the contract and evaluating policies... (Estimated time: ~1-3 minutes)"):
            try:
                # Send HTTP POST to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                
                if response.status_code == 200:
                    report = response.json()
                    violations = report.get("violations", [])
                    total_violations = len(violations)
                    
                    if total_violations == 0:
                        st.success("✅ Contract cleared. Zero policy violations detected.")
                    else:
                        # --- DASHBOARD METRICS ---
                        st.markdown("### 📊 Risk Overview")
                        
                        # Calculate severity counts
                        high_risk = sum(1 for v in violations if v['severity'].lower() == 'high')
                        med_risk = sum(1 for v in violations if v['severity'].lower() == 'medium')
                        low_risk = sum(1 for v in violations if v['severity'].lower() == 'low')
                        
                        # Display KPI Columns
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(label="Total Violations", value=total_violations)
                        col2.metric(label="High Risk 🔴", value=high_risk)
                        col3.metric(label="Medium Risk 🟡", value=med_risk)
                        col4.metric(label="Low Risk 🔵", value=low_risk)
                        
                        # Display Severity Chart
                        st.markdown("#### Violation Breakdown")
                        chart_data = pd.DataFrame(
                            {"Severity": ["High", "Medium", "Low"], "Count": [high_risk, med_risk, low_risk]}
                        )
                        st.bar_chart(chart_data.set_index("Severity"))

                        # --- DETAILED ADJUDICATION ---
                        st.markdown("---")
                        st.markdown("### 🚨 Adjudication Report")
                        
                        for i, violation in enumerate(violations, 1):
                            sev = violation['severity'].upper()
                            emoji = "🔴" if sev == 'HIGH' else "🟡" if sev == 'MEDIUM' else "🔵"
                            
                            with st.expander(f"{emoji} Violation {i} | Severity: {sev}"):
                                # Use columns for a cleaner layout inside the expander
                                v_col1, v_col2 = st.columns([1, 1])
                                
                                with v_col1:
                                    st.markdown("**Violated Internal Policy:**")
                                    st.info(violation['violated_policy'])
                                    
                                    st.markdown("**AI Explanation:**")
                                    st.warning(violation['explanation'])
                                
                                with v_col2:
                                    st.markdown("**Offending Contract Clause:**")
                                    st.code(violation['clause_text'], language="markdown")
                                    
                else:
                    st.error(f"Backend Error [{response.status_code}]: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Fatal Error: Could not connect to backend. Is FastAPI running on port 8000?")
            except Exception as e:
                st.error(f"System failure: {e}")
    else:
        st.warning("You must upload a file before running the analysis.")