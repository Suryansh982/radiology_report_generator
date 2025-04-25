import os
import io
import base64
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import fpdf
import openai
import uuid
import time
from typing import Optional, Dict
import pyperclip

# Load environment variables
load_dotenv()

# Get API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()

# from openai import OpenAI

# Instantiate the new client
client = OpenAI(api_key=api_key)

# Function to encode the image to base64
def encode_image(image_file) -> str:
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

# Function to generate report with OpenAI with forced analysis
def generate_report(image_base64: str, xray_type: str, patient_info: Optional[Dict] = None) -> str:
    # Strong prompt that forces analysis
    prompt = f"""
    You are a senior radiologist with 20 years of experience analyzing medical images. 
    Carefully examine this {xray_type} X-ray image and provide a comprehensive radiology report.
    
    IMPORTANT: You MUST provide a detailed analysis of the image. If the image quality is poor, 
    state that clearly but still attempt an interpretation based on visible features.
    
    Report Structure (MUST FOLLOW):
    
    **CLINICAL INFORMATION**
    {patient_info if patient_info else "Not provided"}
    
    **TECHNIQUE**
    - Digital {xray_type} radiograph
    - Single view (unless multiple views are visible)
    
    **COMPARISON**
    - No prior studies available for comparison (or describe if comparisons exist)
    
    **FINDINGS**
    - Describe all anatomical structures visible
    - Note any abnormalities: opacities, fractures, displacements, etc.
    - Describe normal findings where appropriate
    - Include measurements if relevant (e.g., cardiac silhouette size)
    
    **IMPRESSION**
    - Summary of key findings
    - Differential diagnoses by likelihood
    - Recommendations for follow-up if indicated
    
    Use professional radiology terminology. Be thorough but concise. 
    If absolutely uncertain, state "Nondiagnostic image quality" but still attempt interpretation.
    """
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a highly experienced radiologist."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            report = response.choices[0].message.content
            return report
        except Exception as e:
            if attempt == max_retries - 1:
                print("SOMETHING WENT WRONG:", e)
                st.error(f"Failed to generate report: {str(e)}")
                return "Error generating report."
            time.sleep(retry_delay)

# Function to create professional PDF report with dark text
def create_pdf_report(report_text: str, patient_name: str = "", patient_id: str = "", 
                     doctor: str = "", image_data: Optional[bytes] = None) -> bytes:
    pdf = fpdf.FPDF()
    pdf.add_page()
    
    # Set default text color to black
    pdf.set_text_color(0, 0, 0)  # Black text
    
    # Add hospital/clinic header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "MEDICAL IMAGING CENTER", 0, 1, "C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Department of Radiology", 0, 1, "C")
    pdf.ln(5)
    
    # Report title
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "RADIOLOGY REPORT", 0, 1, "C")
    pdf.ln(5)
    
    # Add report metadata
    pdf.set_font("Helvetica", "", 10)
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    pdf.cell(50, 5, f"Report ID: {report_id}", 0, 0)
    pdf.cell(90, 5, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(3)
    
    # Patient information box with light gray background
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, pdf.get_y(), 190, 20, style='F')
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "PATIENT INFORMATION", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 5, f"Name: {patient_name if patient_name else 'Not provided'}", 0, 0)
    pdf.cell(60, 5, f"ID: {patient_id if patient_id else 'Not provided'}", 0, 1)
    pdf.ln(8)
    
    # Add image if available
    if image_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        # Add image with border
        pdf.set_draw_color(200, 200, 200)
        pdf.rect(pdf.w/2 - 45, pdf.get_y(), 90, 90)
        pdf.image(tmp_path, x=pdf.w/2 - 40, y=pdf.get_y()+5, w=80)
        pdf.ln(95)
        os.unlink(tmp_path)
    
    # Report content - ensure all text is processed
    pdf.set_font("Helvetica", "", 10)
    
    # First, ensure the entire report is included by using multi_cell for the whole content
    # This serves as a fallback if section parsing fails
    pdf.multi_cell(0, 5, report_text)
    
    # Then try to parse sections for better formatting
    try:
        # Process each section
        sections = {
            "CLINICAL INFORMATION": "",
            "TECHNIQUE": "",
            "COMPARISON": "",
            "FINDINGS": "",
            "IMPRESSION": ""
        }
        
        current_section = None
        lines = report_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # Check for section headers
            is_section = False
            for section in sections:
                if line.upper().startswith(section):
                    current_section = section
                    is_section = True
                    i += 1
                    break
                    
            if is_section:
                pdf.ln(5)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, current_section, 0, 1)
                pdf.set_font("Helvetica", "", 10)
            elif current_section:
                # Collect all lines until next section
                section_content = []
                while i < len(lines):
                    line = lines[i].strip()
                    if any(line.upper().startswith(section) for section in sections if section != current_section):
                        break
                    if line:
                        section_content.append(line)
                    i += 1
                
                # Add the content to the PDF
                for content_line in section_content:
                    content_line = content_line.replace("**", "").strip()
                    if content_line:
                        pdf.multi_cell(0, 5, content_line)
                        pdf.ln(2)
    except Exception as e:
        # If section parsing fails, just use the full content that was already added
        pass
    
    # Add signature line
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 5, "Radiologist:", 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, doctor if doctor else "AI Preliminary Interpretation", 0, 1)
    pdf.ln(5)
    
    # Add disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "DISCLAIMER: This AI-generated report is for preliminary informational purposes only and has not been reviewed by a licensed radiologist. It should not be used as the sole basis for clinical decision-making. Final interpretation requires review by a qualified physician.", 0, 1)
    
    return pdf.output(dest="S").encode("latin1")

# Streamlit UI with enhanced features
def main():
    st.set_page_config(
        page_title="AI Radiology Report Generator", 
        layout="wide",
        page_icon="🩺"
    )
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #e9ecef;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stDownloadButton>button {
            background-color: #27ae60;
            color: white;
        }
        .report-box {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: white;
            color: black !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .report-box pre, .report-box code {
            white-space: pre-wrap;
            word-wrap: break-word;
            color: black !important;
        }
        .report-text {
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    _, col2 = st.columns([1, 6])
    with col2:
        st.title("AI Radiology Report Generator")
        st.markdown("Generate professional radiology reports from X-ray images with AI assistance")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Patient & Study Details")
        
        # Study information
        xray_type = st.selectbox(
            "Imaging Study Type",
            ["Chest Xray", "MRI", "Knee Xray", "Other"],
            index=0
        )
        
        study_date = st.date_input("Study Date", datetime.today())
        priority = st.radio("Priority", ["Routine", "Urgent", "STAT"])
        
        # Patient information
        st.subheader("Patient Information")
        patient_name = st.text_input("Full Name")
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Medical Record #")
        with col2:
            patient_dob = st.date_input("Date of Birth", max_value=datetime.today())
        
        col1, col2 = st.columns(2)
        with col1:
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=0)
        with col2:
            patient_gender = st.selectbox("Sex", ["", "Male", "Female", "Other"])
        
        # Clinical information
        clinical_notes = st.text_area("Clinical History/Indication")
        
        # Radiologist information
        st.subheader("Reporting Physician")
        radiologist = st.text_input("Radiologist Name", "Dr. Smith")
        
        # Settings
        st.subheader("AI Settings")
        confidence_level = st.slider("Analysis Confidence Threshold", 1, 10, 7, 
                                   help="Higher values make the AI more conservative in its findings")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.header("Image Upload")
        uploaded_file = st.file_uploader(
            "Upload DICOM or JPEG image", 
            type=["jpg", "jpeg", "png", "dcm"],
            help="For best results, upload high-quality images in standard projections"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image with enhancements
            try:
                image = Image.open(uploaded_file)
                
                # Convert to grayscale if not already
                if image.mode != 'L':
                    image = image.convert('L')
                
                # Enhance contrast
                st.image(
                    image, 
                    caption=f"Uploaded {xray_type} Image - {priority} Priority",
                    use_column_width=True,
                    clamp=True
                )
                
                # Generate report button
                if st.button("Generate Radiology Report", type="primary"):
                    with st.spinner("Analyzing image with AI... This may take 20-30 seconds"):
                        # Prepare patient info
                        patient_info = {
                            "Name": patient_name,
                            "MRN": patient_id,
                            "DOB": patient_dob.strftime("%Y-%m-%d") if patient_dob else "",
                            "Age": patient_age,
                            "Sex": patient_gender,
                            "Clinical Notes": clinical_notes,
                            "Study Date": study_date.strftime("%Y-%m-%d"),
                            "Priority": priority
                        }
                        
                        # Format patient info string
                        patient_info_str = "\n".join(
                            f"{k}: {v}" for k, v in patient_info.items() if v
                        )
                        
                        # Reset file pointer and encode image
                        uploaded_file.seek(0)
                        base64_image = encode_image(uploaded_file)
                        
                        # Generate report
                        start_time = time.time()
                        report = generate_report(
                            base64_image, 
                            xray_type, 
                            patient_info=patient_info_str
                        )
                        processing_time = time.time() - start_time
                        
                        if report:
                            # Store report in session state
                            st.session_state.report = report
                            st.session_state.image_data = uploaded_file.getvalue()
                            st.session_state.patient_name = patient_name
                            st.session_state.patient_id = patient_id
                            st.session_state.radiologist = radiologist
                            st.session_state.processing_time = processing_time
                            st.session_state.patient_info = patient_info_str
                            
                            # Indicate success
                            st.success(f"Report generated in {processing_time:.1f} seconds")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.header("Radiology Report")
        
        # Check if report exists in session state
        if 'report' in st.session_state:
            # Fix for f-string issue
            formatted_report = st.session_state.report.replace('\n', '<br>')
            
            # Display in a nice box with dark text
            st.markdown(st.session_state.report)

            
            try:
                # PDF download button
                pdf_data = create_pdf_report(
                    st.session_state.report,
                    patient_name=st.session_state.patient_name,
                    patient_id=st.session_state.patient_id,
                    doctor=st.session_state.radiologist,
                    image_data=st.session_state.image_data
                )
                
                st.download_button(
                    label="Download Formal PDF Report",
                    data=pdf_data,
                    file_name=f"RAD_{xray_type.replace(' ', '_')}_{patient_name or 'Unknown'}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    help="Formal PDF report suitable for medical records"
                )
                
                # Add copy button
                st.button("Copy Report Text", 
                        on_click=lambda: pyperclip.copy(st.session_state.report) if 'pyperclip' in globals() else st.session_state.report,
                        help="Copy report text to clipboard")
                
            except Exception as e:
                st.error(f"Error creating PDF: {str(e)}")
                # Fallback - show raw text that can be manually copied
                st.text_area("Report Text", st.session_state.report, height=300)
        else:
            with st.container():
                st.info("""
                **Instructions:**
                1. Upload an X-ray image (JPEG, PNG)
                2. Fill in patient and study details
                3. Click "Generate Radiology Report"
                
                The AI will provide a preliminary interpretation following standard radiology report structure.
                """)

# Run the app
if __name__ == "__main__":
    main()