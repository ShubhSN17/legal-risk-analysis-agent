import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="Legal Risk Agent", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Risk Analysis Agent")
st.markdown("Upload a third-party contract. The AI will cross-reference the text against our internal company policies and flag any risks.")

# File Uploader
uploaded_file = st.file_uploader("Choose a PDF contract", type="pdf")

if st.button("Analyze Contract"):
    if uploaded_file is not None:
        with st.spinner("Agent is reading the contract and querying the vector database... (This may take a minute)"):
            try:
                # Send the file via HTTP POST to our FastAPI backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                # Make sure your FastAPI server is running on port 8000!
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)
                
                if response.status_code == 200:
                    report = response.json()
                    
                    total = report.get('total_violations', 0)
                    if total == 0:
                        st.success("✅ Contract cleared. No policy violations found.")
                    else:
                        st.error(f"🚨 Analysis Complete! Found {total} policy violation(s).")
                    
                    # Display the violations beautifully
                    for i, violation in enumerate(report.get("violations", []), 1):
                        # Color code based on severity
                        emoji = "🔴" if violation['severity'] == 'High' else "🟡" if violation['severity'] == 'Medium' else "🔵"
                        
                        with st.expander(f"{emoji} Violation {i}: {violation['severity']} Risk"):
                            st.markdown(f"**Violated Company Policy:** {violation['violated_policy']}")
                            st.markdown(f"**Offending Contract Clause:**\n> `{violation['clause_text']}`")
                            st.markdown(f"**AI Adjudication:** {violation['explanation']}")
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is your FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload a PDF first.")