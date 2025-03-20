import streamlit as st
import pandas as pd
import numpy as np
import re
import random
import pickle
from fpdf import FPDF
from pycaret.classification import load_model 
import google.generativeai as genai 

# PAGE SET UP
# ===========
## Initialize our streamlit app
st.set_page_config(
    page_title="Leak Detection", 
    page_icon='✈️',
    layout='wide'
    )

api_key_goog = st.secrets["general"]["goog_api_key"]
genai.configure(api_key=api_key_goog)

# Prompt generation function
def generate_prompt(input):
    prompt = f"You are the best engineer, mechanic, or technical person at Airbus who is an expert in fuel leak prevention. \
    Right now, leaks have been detected on some flights. Given the following context, give the user some recommendations on what to do next. \
    Keep your answers short, simple, and to the point. If no input is given, make generic recommendations"
    return prompt

# Function to call the model and get responses
def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if input != "":
        prompt = generate_prompt(input)
        response = model.generate_content([prompt, input])
    else:
        prompt = generate_prompt("")
        response = model.generate_content(prompt)
    
    return response.text

# FORMATTING
# ==========

# Set custom CSS styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@800&display=swap');
    
    .header {
        font-family: 'Open Sans', sans-serif;
        font-size: 48px;
        font-weight: 600 !important;
        color: #333;
        margin-top: 50px;
    }

    .body {
        font-family: 'Montserrat', sans-serif;
        font-size: 20px;
        font-weight: 400 !important;
        color: #555;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# File Uploaders

# BUTTON CONFIGURATION
# ====================
st.markdown("""
    <style>
    .stButton > button {
        height: auto;
        padding-top: 20px;
        padding-bottom: 20px;
        font-weight: bold !important;
        color: white; /* white text in normal state */
        border: 5px solid black; /* black border */
        border-radius: 40px;
        width: 100%;
        cursor: pointer;
        background-color: black; /* black background in normal state */
    }
    .stButton > button:hover {
        background-color: white; /* white background on hover */
        color: black; /* pink text on hover */
        border: 5px solid black; /* Ensures border stays pink */
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS to style buttons and radio buttons
st.markdown("""
    <style>
    /* Style for buttons */
    .stButton > button {
        height: auto;
        padding-top: 20px;
        padding-bottom: 20px;
        font-weight: bold !important;
        color: white;
        border: 5px solid black;
        border-radius: 40px;
        width: 100%;
        cursor: pointer;
        background-color: black;
    }
    .stButton > button:hover {
        background-color: white;
        color: black;
        border: 5px solid black;
    }

    /* Style for radio buttons */
    div[data-testid="stRadio"] label {
        display: block;
        font-weight: bold;
        background-color: black;
        color: black;
        padding: 15px;
        border-radius: 40px;
        border: 3px solid black;
        margin-bottom: 5px; /* Spacing between options */
        cursor: pointer;
        text-align: center;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# LOADING MODEL
# =============
with st.spinner("Loading model..."):
    model = load_model("model_f2") 

# WEBSITE HEADER
# ==============
st.image("images/title.png", use_container_width=True)
st.write('---')

# NAVIGATION BUTTONS
# ==================

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("HOME"):
        st.session_state.page = "HOME"    
with col2:
    if st.button("DATASET UPLOAD"):
        st.session_state.page = "DATASET UPLOAD" 
with col3:
    if st.button("RECOMMENDATIONS"):
        st.session_state.page = "RECOMMENDATIONS" 

menu = ["HOME", "DATASET UPLOAD", "RECOMMENDATIONS"]
# Initialize session state if not set

if 'page' not in st.session_state:
    st.session_state.page = "HOME"

# Show content based on the selected page
page = st.session_state.page


# HOME PAGE
# =========

if page == "HOME":
    st.write('')
    st.image('images/a400m.png')
    st.write('')
    st.write('')
    st.write('')
    st.image('images/home1.png')
    st.write('')
    st.write('')
    st.write('')
    st.image('images/predfuelleaks.png')
    st.write('')
    st.write('')
    st.write('')
    st.image('images/causes.png')
    st.write('---')    
    st.markdown("""
        <p style="font-family: \'Montserrat\', sans-serif; font-size: 20px; font-weight: 400; text-align: center">
        This web application is part of the Master's in Business Analytics and Big Data Corporate Project.</p>
        """, unsafe_allow_html=True)


# DATASET UPLOAD PAGE
# ===================

if page == "DATASET UPLOAD":
    st.write('')
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/upload.png')
    with col2:
        st.image('images/upload2.png')
        uploaded_file = st.file_uploader("", type=["csv"])
        st.write('')


# Backend Stuff
# -------------

    # Expected columns for schema validation
    expected_columns = [
        "FUEL_USED_2", "FUEL_USED_3", "FUEL_USED_4", "FW_GEO_ALTITUDE", "VALUE_FOB",
        "VALUE_FUEL_QTY_CT", "VALUE_FUEL_QTY_FT1", "VALUE_FUEL_QTY_FT2", "VALUE_FUEL_QTY_FT3",
        "VALUE_FUEL_QTY_FT4", "VALUE_FUEL_QTY_LXT", "VALUE_FUEL_QTY_RXT", "FLIGHT_PHASE_COUNT",
        "FUEL_USED_1", "Flight", "MSN", "UTC_TIME"
    ]

    msn_number = "##"  # Default MSN placeholder
    leaks_detected = []  # Initialize variable

    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter =";")  # Load in smaller parts
        #df = pd.concat(df)  # Combine chunks after processing

        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        # Extract MSN number from file name
        match = re.search(r"msn_(\d{2})", uploaded_file.name)
        if match:
            msn_number = match.group(1)
        else:
            st.warning("MSN code could not be extracted, set to default ##")
        
        if not missing_columns:
            st.success("Schema Verified")
        else:
            st.error(f"Incorrect Schema for Model. Verify features == {missing_columns}")

        # Preprocessing logic
        df['NEW_FLIGHT'] = df.groupby('Flight')['FLIGHT_PHASE_COUNT'].diff().lt(0)
        df['FLIGHT_INSTANCE'] = df.groupby('Flight')['NEW_FLIGHT'].cumsum()
        df['FLIGHT_ID'] = df['Flight'].astype(str) + "_" + df['FLIGHT_INSTANCE'].astype(str)
        df['START_FOB'] = df.groupby('FLIGHT_ID')['VALUE_FOB'].transform('first')
        df = df[df["FLIGHT_PHASE_COUNT"] == 8.0]
        df = df.sort_index()
        df['TOTAL_FUEL_USED'] = df['FUEL_USED_1'] + df['FUEL_USED_2'] + df['FUEL_USED_3'] + df['FUEL_USED_4']
        df['EXPECTED_FOB'] = df['START_FOB'] - df['TOTAL_FUEL_USED']
        df["FOB_DIFFERENCE"] = (df['EXPECTED_FOB'] - df['VALUE_FOB']).abs()
        df['FOB_CHANGE'] = df['VALUE_FOB'].diff().fillna(0)
        df['EXPECTED_FOB_CHANGE'] = -df['TOTAL_FUEL_USED'].diff().fillna(0)
        df['FUEL_LEAK_RATE'] = df['EXPECTED_FOB_CHANGE'] - df['FOB_CHANGE']
        df['TOTAL_FUEL_LW'] = df['VALUE_FUEL_QTY_LXT'] + df['VALUE_FUEL_QTY_FT1'] + df['VALUE_FUEL_QTY_FT2']
        df['TOTAL_FUEL_RW'] = df['VALUE_FUEL_QTY_RXT'] + df['VALUE_FUEL_QTY_FT3'] + df['VALUE_FUEL_QTY_FT4']
        df['LW_RW_DIFF'] = (df['TOTAL_FUEL_LW'] - df['TOTAL_FUEL_RW']).abs()
        df['FUEL_IN_TANKS'] = df['VALUE_FUEL_QTY_CT'] + df['VALUE_FUEL_QTY_FT1'] + df['VALUE_FUEL_QTY_FT2'] + df['VALUE_FUEL_QTY_FT3'] + df['VALUE_FUEL_QTY_FT4'] + df['VALUE_FUEL_QTY_LXT'] + df['VALUE_FUEL_QTY_RXT']
        df['CALC_VALUE_FOB_DIFF'] = df['FUEL_IN_TANKS'] - df['VALUE_FOB']
        df['START_FOB_VS_FOB_FUELUSED'] = df['START_FOB'] - (df['FUEL_IN_TANKS'] + df['TOTAL_FUEL_USED'])
        df['ALTITUDE_DIFF'] = df['FW_GEO_ALTITUDE'].diff().fillna(0)
        df['UTC_TIME'] = pd.to_datetime(df['UTC_TIME'])


        # Save the original Flight column    
        df = df.sort_values(by=['FLIGHT_ID', 'UTC_TIME']).reset_index(drop=True)
        flight_reference = df['Flight'].reset_index(drop=True)  # Preserve flight reference
        
        options = [
            "Run on Entire Dataset",
            "Run on a Random Flight",
            "Select a Specific Flight"
        ]

        st.markdown("""
        <h5 style="font-family: \'Open Sans\', sans-serif; font-size: 48px; font-weight: 800; text-align: center">
                 Select Model Execution Option</h1>
            """, unsafe_allow_html=True)    

        # Create columns for horizontal layout
        col1, col2, col3 = st.columns(len(options))

        # Initialize session state for selection
        if "selected_option" not in st.session_state:
            st.session_state.selected_option = options[0]  # Default selection

        # Function to update selection
        def select_option(option):
            st.session_state.selected_option = option

        st.markdown("""
        <style>
        .custom-radio {
            font-weight: bold;
            padding: 15px 20px;
            border-radius: 40px;
            border: 3px solid black;
            text-align: center;
            cursor: pointer;
            transition: 0.3s;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .selected {
            background-color: black !important;
            color: white !important;
        }
        .unselected {
            background-color: black;
            color: white;
        }
        .custom-radio:hover {
            background-color: white;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)         

        # Display buttons in columns
        for col, option in zip([col1, col2, col3], options):
            with col:
                btn_class = "selected" if st.session_state.selected_option == option else "unselected"
                if st.button(option, key=option, help="Click to select", use_container_width=True):
                    select_option(option)

        flight_code = None

        if st.session_state.selected_option == "Select a Specific Flight":

            st.markdown(f"""
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
            <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
            Select Specifc Flight Selected Flight Selected
            </h6>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
            <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: normal; text-align: center;">
            Select Flight
            </h6>
            """, unsafe_allow_html=True)
            flight_code = st.selectbox("", df["Flight"].unique())

            st.write('')
            st.write('')

            # Filter dataset for selected flight
            df_filtered = df[df["Flight"] == flight_code].copy()  # Use .copy() to avoid modifying original df

            # Ensure FLIGHT_ID exists
            if "FLIGHT_ID" not in df_filtered.columns:
                st.error("FLIGHT_ID column is missing!")
            else:
                flight_reference = df_filtered[['Flight', 'FLIGHT_ID']].reset_index(drop=True)

            # Prepare numerical features for model
            df_sorted = df_filtered.drop(columns=['FLIGHT_PHASE_COUNT', 'Flight', 'FLIGHT_INSTANCE'], errors='ignore')
            df_sorted = df_sorted.select_dtypes(include=[np.number])

            if st.button("Detect for Leaks"):
                # Run model prediction
                predictions = model.predict(df_sorted)
                probabilities = model.predict_proba(df_sorted)

                # Create result DataFrame
                result_df = pd.DataFrame({
                    "Leak_Detected": predictions,
                    "Leak_Probability": probabilities[:, 1]  # Get probability of leak (Class 1)
                })

                # Merge back flight details
                result_df = pd.concat([flight_reference, result_df], axis=1)
                result_df = result_df[result_df["Leak_Detected"] == 1]

                # Identify leaks
                leaks_detected = result_df.loc[result_df["Leak_Detected"] == 1, "Flight"].unique().tolist()
                if leaks_detected:
                    st.write('')
                    st.write('')
                    cola, colb = st.columns(2)
                    with cola:
                        st.write('')
                        st.write('')
                        st.image('images/leakleft.png')
                        flight_ids_with_leaks = ', '.join(result_df['FLIGHT_ID'].unique().astype(str))
                        col1, col2, col3 = st.columns([1,10,1])
                        with col1:
                            st.empty()
                        with col2:
                            st.markdown(f"""
                                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                                <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
                                {flight_ids_with_leaks}
                                </h6>
                                """, unsafe_allow_html=True)
                        with col3:
                            st.empty()  
                    with colb:
                        st.image('images/leakright.png')

                else:
                    st.success("No Leaks Detected")

                st.write('')
                st.write('---')
                st.write('')
                st.markdown(f"""
                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                <h3 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
                Dataframe of flights with leaks
                </h3>
                """, unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)

        elif st.session_state.selected_option == "Run on a Random Flight":

            st.markdown(f"""
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
            <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
            Run on a Random Flight Selected
            </h6>
            """, unsafe_allow_html=True)

            # Check if a random flight has already been selected in the session state
            if 'flight_code' not in st.session_state:
                # If not, select a random flight and store it in session state
                st.session_state.flight_code = random.choice(df["Flight"].unique())
            
            flight_code = st.session_state.flight_code
            st.markdown(f"""
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
            <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: normal; text-align: center;">
            Random Flight Selection: {flight_code}
            </h6>
            """, unsafe_allow_html=True)
            
            # Filter dataset for selected flight
            df_filtered = df[df["Flight"] == flight_code].copy()  # Use .copy() to avoid modifying original df

            # Ensure FLIGHT_ID exists
            if "FLIGHT_ID" not in df_filtered.columns:
                st.error("FLIGHT_ID column is missing!")
            else:
                flight_reference = df_filtered[['Flight', 'FLIGHT_ID']].reset_index(drop=True)

            # Prepare numerical features for model
            df_sorted = df_filtered.drop(columns=['FLIGHT_PHASE_COUNT', 'Flight', 'FLIGHT_INSTANCE'], errors='ignore')
            df_sorted = df_sorted.select_dtypes(include=[np.number])

            if st.button("Detect for Leaks"):
                # Run model prediction
                predictions = model.predict(df_sorted)
                probabilities = model.predict_proba(df_sorted)

                # Create result DataFrame
                result_df = pd.DataFrame({
                    "Leak_Detected": predictions,
                    "Leak_Probability": probabilities[:, 1]  # Get probability of leak (Class 1)
                })

                # Merge back flight details
                result_df = pd.concat([flight_reference, result_df], axis=1)
                result_df = result_df[result_df["Leak_Detected"] == 1]

                # Identify leaks
                leaks_detected = result_df.loc[result_df["Leak_Detected"] == 1, "Flight"].unique().tolist()
                if leaks_detected:
                    st.write('')
                    st.write('')
                    cola, colb = st.columns(2)
                    with cola:
                        st.write('')
                        st.write('')
                        st.image('images/leakleft.png')
                        flight_ids_with_leaks = ', '.join(result_df['FLIGHT_ID'].unique().astype(str))
                        col1, col2, col3 = st.columns([1,10,1])
                        with col1:
                            st.empty()
                        with col2:
                            st.markdown(f"""
                                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                                <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
                                {flight_ids_with_leaks}
                                </h6>
                                """, unsafe_allow_html=True)
                            if st.button("See Recommendations"):
                                st.session_state.page = "RECOMMENDATIONS" 
                        with col3:
                            st.empty()  
                    with colb:
                        st.image('images/leakright.png')
                else:
                    st.success("No Leaks Detected")

                st.write('')
                st.write('---')
                st.write('')
                st.markdown(f"""
                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                <h3 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
                Dataframe of flights with leaks
                </h3>
                """, unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)

        elif st.session_state.selected_option == "Run on Entire Dataset":
            
            st.markdown(f"""
            <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
            <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
            Run on Entire Dataset Selected
            </h6>
            """, unsafe_allow_html=True)

            # Drop Flight-related columns after selection
            df_sorted = df.drop(columns=['FLIGHT_PHASE_COUNT', 'Flight', 'FLIGHT_INSTANCE'])
            df_sorted = df_sorted.select_dtypes(include=[np.number])
            
            if st.button("Detect for Leaks"):
                # Run model prediction
                predictions = model.predict(df_sorted)
                probabilities = model.predict_proba(df_sorted)
                
                result_df = pd.DataFrame(predictions, columns=["Leak_Detected"])
                result_df = pd.DataFrame({
                "Leak_Detected": predictions,
                "Leak_Probability": probabilities[:, 1]  # Get probability of leak (Class 1)
                })
                
                result_df = pd.concat([flight_reference, df[['FLIGHT_ID']], result_df], axis=1)
                result_df = result_df[result_df["Leak_Detected"] == 1]

                leaks_detected = result_df.loc[result_df["Leak_Detected"] == 1, "Flight"].unique().tolist()
                if leaks_detected:
                    st.write('')
                    st.write('')
                    cola, colb = st.columns(2)
                    with cola:
                        st.write('')
                        st.write('')
                        st.image('images/leakleft.png')
                        flight_ids_with_leaks = ', '.join(result_df['FLIGHT_ID'].unique().astype(str))
                        col1, col2, col3 = st.columns([1,10,1])
                        with col1:
                            st.empty()
                        with col2:
                            st.markdown(f"""
                                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                                <h6 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
                                {flight_ids_with_leaks}
                                </h6>
                                """, unsafe_allow_html=True)
                        with col3:
                            st.empty()  
                    with colb:
                        st.image('images/leakright.png')

                else:
                    st.success("No Leaks Detected")

                st.write('')
                st.write('---')
                st.write('')
                st.markdown(f"""
                <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                <h3 style="color: Black; font-family: 'Montserrat', sans-serif; font-weight: bold; text-align: center;">
                Dataframe of flights with leaks
                </h3>
                """, unsafe_allow_html=True)
                st.dataframe(result_df, use_container_width=True)

                # Generate report option
                # st.header("Generate Report")
                # if st.button("Generate PDF Report"):
                #     class PDF(FPDF):
                #         def header(self):
                #             self.set_font("Arial", "B", 12)
                #             self.cell(0, 10, "Fuel Leak Detection Report", ln=True, align="C")
                #             self.ln(10)

                #     pdf = PDF()
                #     pdf.add_page()
                #     pdf.set_font("Arial", size=12)
                #     pdf.cell(200, 10, f"Aircraft MSN: {msn_number}", ln=True, align="L")
                #     if flight_code:
                #         pdf.cell(200, 10, f"Flight Number: {flight_code}", ln=True, align="L")
                #     pdf.cell(200, 10, f"Leak Detection Result: {'Leak Detected' if leaks_detected else 'No Leak Detected'}", ln=True, align="L")
                #     report_filename = "fuel_leak_report.pdf"
                #     pdf.output(report_filename)
                #     st.success("Report Generated Successfully")
                #     with open(report_filename, "rb") as file:
                #         st.download_button("Download Report", file, file_name=report_filename)


# DATASET UPLOAD PAGE
# ===================

if st.session_state.page == "RECOMMENDATIONS":

    st.write('')
    st.write('')
    st.write('')
    st.image('images/recommendations.png')
    st.write('---')

    col1, col2 = st.columns(2)
    with col1:
        input = st.text_area("", placeholder="Describe the overall maintenance history and/or known issues of these flights or general overall context:", height=200)
        #input = st.text_input("Describe the overall maintenance history and/or known issues of these flights or general overall context: ", key="input")
        submit = st.button("Get recommendations")
    with col2:
        st.image('images/genai.png')

    if submit:
        st.write('---')
        if input != "":
            response = get_gemini_response(input)  # Use the provided input to generate the response
        else:
            response = get_gemini_response("")  # Use default context if input is empty

        # Store the response in session state
        st.session_state.response = response

        # Display the response
        st.subheader("The AI's Analysis and Recommendations:")
        st.write(response)


