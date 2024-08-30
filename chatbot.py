import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

import pandas as pd 

# Load environment settings
#from dotenv import load_dotenv
#load_dotenv()

# Load OpenAI API key

# Initialize the model
llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"],model="gpt-4o-mini", temperature=0.5, max_tokens=1200)









# Define scenarios
scenarios = [
    {
        "id": 1,
        "description": ("A 62-year-old man presented to the Neurology Clinic of KFSH with non-disabling resting tremors in both hands. "
                        "He experienced mild gradual progressive stiffness of his body associated with slow movements, slurred speech, and "
                        "postural instability. He found difficulty in the initiation and termination of voluntary movements. Although the "
                        "patient had a hand tremor that disappeared on writing, his calligraphy became slower and looked cramped and smaller "
                        "than before. The condition started with resting tremors in his left hand for one year, which has progressed to involve "
                        "both hands. The patient was depressed because of his illness; however, he denied any history of taking medications for that. "
                        "Family history revealed that his grandparents had similar tremors later in their lives."),
        "tests": {
            "images": [
                {"path": "data/scenario_1/image_1.png", "caption": "MRI Image Example"}
            ],
            "vital_signs": {
                "BP": "130/85",
                "Pulse": "82/min",
                "Respiratory rate": "18/min",
                "Temperature": "37⁰C"
            },
            "neurological_examination": ("Face: was expressionless face with normal cognition. \n\n"
                                         "Gait: the patient tends to lean forward while walking with small quick steps & reduced swinging of the arms. \n\n"
                                         "Motor examination: there were intermittent mild resting pill rolling tremors observed in both hands with mild signs of asymmetrical cogwheel rigidity; the left side was more affected than the right one. \n\n"
                                         "Sensory examination: unremarkable."),
            "general_examination": ("Consciousness level: The patient was conscious and oriented to time, place, and people. \n\n"
                                    "Chest and cardiac examination: unremarkable. \n\n"
                                    "Genitourinary examination: revealed a prostatic enlargement."),
            "management_plan": ("The consultant started his plan of management by prescribing rasagiline 1mg/day orally. "
                                "He requested a PET scan of the brain, and its results are shown in the attached image (B) compared with the finding of a normal person in image (A): see in the next page."),
            "lab_results": {
                "Hb": "16 g/dL",
                "RBC count": "8 x 10⁶/mm³",
                "Haematocrit": "60%",
                "PaO₂": "53 mm/Hg (75 - 100 mmHg)",
                "PaCO₂": "43 mm/Hg (35 - 45 mmHg)",
                "Oxygen saturation": "Low"
            }
        },
        "expected_diagnosis": "Parkinson's disease"
    },
    {
        "id": 2,
        "description": ("Ghada, a 2-year-old girl, was brought to the Pediatric Outpatient Clinic by her mother. She told the doctor that 'my girl was born with a normal weight and was all right for the first one and a half years, but for the past four to five months, I noticed that she became increasingly short of breath and her lips and tongue became blue whenever she played with her cousins or crying for any reason. She would then sit in the squatting position for some time until she recovered' (Figure 1). There was no family history of a similar condition, history of infection, or medications during pregnancy."),
        "tests": {
            "images": [
                {"path": "data/scenario_2/image_1.png", "caption": "Image 1"},
                {"path": "data/scenario_2/image_2.png", "caption": "Image 2"},
                {"path": "data/scenario_2/image_3.png", "caption": "ECG Image"},
                {"path": "data/scenario_2/image_4.png", "caption": "Chest X-ray"}
            ],
            "vital_signs": {
                "Pulse": "140/min",
                "Blood pressure": "102/64 mmHg",
                "Respiratory rate": "30/min",
                "Temperature": "36.5⁰C; measured rectally"
            },
            "general_examination": ("The girl was ill-looking and smaller than expected for her age. \n"
                                    "There was bluish discoloration of the lips and tongue with clubbing of her nails (Fig. 2)."),
            "local_examination": ("Inspection: Diffuse precordial pulsations more marked on the left parasternal area. \n"
                                  "Auscultation: Pansystolic murmur over 3rd and 4th left parasternal spaces with palpable thrill. A single 2nd heart sound was heard on the left 2nd space."),
            "lab_findings": ("ECG (Figure 3) showed right axis deviation, abnormal R in V1 & abnormal T wave and ST segment in V2 and V3.\n"
                             "Echocardiography revealed right ventricular hypertrophy, a large VSD, overriding aorta and narrowing of the pulmonary outlet.\n"
                             "Chest X-ray: Boot shaped heart (Figure 4).\n"),
            "lab_results": {
                "Hb": "16 g/dL",
                "RBC count": "8 x 10⁶/mm³",
                "Haematocrit": "60%",
                "PaO₂": "53 mm/Hg (75 - 100 mmHg)",
                "PaCO₂": "43 mm/Hg (35 - 45 mmHg)",
                "Oxygen saturation": "Low"
            }
        },
        "expected_diagnosis": "Tetralogy of Fallot"
    },
    {
        "id": 3,
        "description": ("Nora, a 60-year-old woman presented to the Rheumatology Clinic with pain and stiffness of the joints of her hands and wrists for the past 2 months. "
                        "The pain and stiffness last for at least an hour in the morning but improve throughout the day and with exercise. "
                        "Similar complaints were reported in her neck and shoulders, which aggravated her life, and she mentioned that looking down worsened the pain. "
                        "Nora also complained of generalized weakness, fatigue, and weight loss during this period. She denied any history of fever, skin rash, vision changes, or photophobia. "
                        "She was prescribed some analgesics by a local doctor, but discontinued them due to the development of epigastric pain."),
        "tests": {
            "images": [
                {"path": "data/scenario_3/image_1.png", "caption": "Elbow Examination"},
                {"path": "data/scenario_3/image_2.png", "caption": "Hand Examination"},
                {"path": "data/scenario_3/image_3.png", "caption": "X-ray of the left wrist"},
                {"path": "data/scenario_3/image_4.png", "caption": "X-ray of the left hand"}
            ],
            "vital_signs": {
                "Temperature": "38°C",
                "Respiratory rate": "18/min",
                "Pulse": "88/min",
                "Blood pressure": "135/90 mmHg"
            },
            "physical_examination": ("The patient was in mild distress. \n"
                                     "Examination of the neck showed painful flexion, low range of motion, and no change in muscle mass or strength. \n\n"
                                     "Examination of the shoulders revealed normal muscle mass and strength. \n\n"
                                     "Examination of the elbows (Figure 1) demonstrated firm, non-tender subcutaneous nodules. \n\n"
                                     "Examination of the hands (Figure 2) showed weak hand grip, painful movements of the fingers and wrists, decreased range of motion, hotness, tenderness, and swelling over the PIPs, MCPs, and wrist joints of both sides. A swan neck deformity was observed, particularly in her left middle and little fingers."),
            "lab_results": {
                "Haemoglobin": "9.5 g/dL",
                "ESR": "85 mm/1st hr",
                "WBCs": "11,500 /μL",
                "Serum Uric acid": "5.5 mg/dL",
                "Antinuclear antibodies (ANA)": "Negative",
                "RA factor": "Positive",
                "Anti-citrullinated peptide antibodies (ACPA)": "Positive"
            },
            "radiological_findings": ("Plain X-ray of the neck: showed that all the cervical vertebrae were normal. \n\n"
                                      "Plain X-ray of the left wrist (AP view) Figures 3: demonstrated subtle diffuse osteopenia of the carpal bones and prominent soft tissue nodule overlying the styloid process (arrow). \n\n"
                                      "Plain X-ray of the left hand (AP view) Figures 4: showed osteopenia around the metacarpophalangeal joints (arrows) with mild soft tissue swelling (arrowheads). \n\n"
                                      "The doctor ordered aspiration from the left wrist joint, which was rich in neutrophils and protein, and negative for crystals and microbial growth. He requested MRI of the spine, and its results were pending.")
        },
        "expected_diagnosis": "Rheumatoid Arthritis"
    },
    {
        "id": 4,
        "description": ("Ali, 75-years-old diabetic man brought by his son to KFSH with spontaneous bleeding from his nose, severe headache and confusion. "
                        "His son mentioned that his father was complaining of generalized weakness and swelling of his ankles and feet for the past month. He also added that his father suffered from generalized itching, decreased urine output and occasional vomiting. "
                        "Ali’ son declared that his father had a history of hypertension and diabetes mellitus for 25 years. Two years ago, he was brought to the Emergency Department because of an acute urine retention, where a senior resident failed to evacuate the urine and recommended an urgent surgical intervention, however, Ali refused. He was not well adherent to his medications."),
        "tests": {
            "images": [
                {"path": "data/scenario_4/image_1.png", "caption": "Pathological Finding 1"},
                {"path": "data/scenario_4/image_2.png", "caption": "Pathological Finding 2"},
                {"path": "data/scenario_4/image_3.png", "caption": "EKG"},
                {"path": "data/scenario_4/image_4.png", "caption": "Abdomino-Pelvic Ultrasound"},
                {"path": "data/scenario_4/image_5.png", "caption": "Dialysis Femoral Catheter"},
                {"path": "data/scenario_4/image_6.png", "caption": "Renal Nuclear Scan: DTPA Test"}
            ],
            "vital_signs": {
                "BP": "200/110 mmHg",
                "Pulse": "95 beat/min",
                "Respiratory rate": "22/min",
                "Temperature": "37 °C"
            },
            "physical_examination": ("Patient was confused, very pale & his breathing was shallow with bad smell. \n\n"
                                    "Skin showed hyperpigmentation and scratch marks (Figs. 1 and 2). \n\n"
                                    "Bilateral pitting edema of the ankles and feet were observed. \n\n"
                                    "Examination of other systems was unremarkable."),
            "lab_results": {
                "BUN": "45 mg/dL",
                "Serum creatinine": "6 mg/dL",
                "Serum Na+": "132 mEq/L",
                "Serum K+": "5.5 mEq/L",
                "PaCO2": "20 mmHg",
                "Serum HCO3-": "16 mmol/L",
                "pH": "7.2",
                "eGFR": "8 mL/min/1.73 m2",
                "Bleeding time": "12 minutes",
                "Hb": "7.0 g/dL",
                "Platelets": "130 x10³/µL",
                "WBC": "5 x 10³/ µL",
                "HbA1c%": "10.8%",
                },
            "lab_findings": ("EKG (Figure 3): shows normal sinus rhythm with hyperacute T-wave in anterior leads. \n\n"
                            "Abdomino-Pelvic Ultrasound (Figure 4) showed reduced renal length and cortical thickness, increased renal cortical echogenicity and poor visibility of the renal pyramids and the renal sinus. \n\n"
                            "After blood cross matching, one unit of blood and platelet were transfused. A dialysis femoral catheter was then inserted (Picture 5) and nephrology consultant was called in. \n\n"
                            "Renal nuclear scan: DTPA Test (Figure 6). The scan from 0 second till 28 minutes showed reduced renal size on both sides as well as renal uptake and excretion of trace; which was more on the right side.")
        },
        "expected_diagnosis": "Chronic Renal Failure"
    },
    {
        "id": 5,
        "description": ("Naser, a 25-year-old man was brought to the emergency department after a road traffic accident complaining of severe chest pain. "
                        "The primary survey of the patient revealed the following:\n"
                        "• The patient was drowsy with an altered level of consciousness.\n"
                        "• Airway: was patent and protected.\n"
                        "• Breathing: severe dyspnea and congested neck veins were reported."),
        "tests": {
            "images": [
                {"path": "data/scenario_5/image_1.png", "caption": "Initial Examination"},
                {"path": "data/scenario_5/image_2.png", "caption": "Chest X-ray (CXR)"},
                {"path": "data/scenario_5/image_3.png", "caption": "Intercostal Chest Tube Insertion 1"},
                {"path": "data/scenario_5/image_4.png", "caption": "Intercostal Chest Tube Insertion 2"}
            ],
            "vital_signs": {
                "Pulse": "130/min",
                "Respiratory rate": "30/min",
                "Blood pressure": "75/40 mmHg",
                "Temperature": "37°C"
            },
            "physical_examination": ("Inspection: showed reduced expansion and decreased movement of the right side of the chest. \n\n"
                                    "Palpation: revealed tenderness over the right side of anterior wall of chest wall, shifting of the apex beat to the left beyond the midclavicular line as well as the trachea was shifted to the left side.\n"
                                    "Percussion: hyper-resonance on the right side of the chest was evident. \n\n"
                                    "Auscultation: no breath sounds were heard on the right side of the patient’s chest.\n\n"
                                    "O2 therapy was started immediately. The pulmonologist requested an urgent CXR; but urgent needle decompression was done before by inserting a needle into the right side of the chest at the second intercostal space in the midclavicular line (Figure 1). Following this procedure, the general condition of the patient improved dramatically."),
            "radiological_findings": ("Chest X-ray (CXR) demonstrated the following (Figure 2):\n\n"
                                    "• Radiolucent hyperexpanded right hemithorax with an absence of the broncho-vascular markings.\n\n"
                                    "• Marked widening of the right-sided intercostal spaces.\n\n"
                                    "• Shifting of the mediastinal structures toward the left side.\n\n"
                                    "• Depression of the right hemidiaphragm.\n\n"
                                    "• Clear right costophrenic angle.\n\n"
                                    "An intercostal chest tube was inserted under local anesthesia at the intersection of the right 5th intercostal space at mid-axillary line, while the other end of the tube was connected to an underwater seal (Figures 3 & 4).")
        },
        "expected_diagnosis": "Pneumothorax"
    },
    {
        "id": 6,
        "description": ("Norah, a 60-year-old woman presented to the emergency department with severe abdominal pain, nausea, and vomiting. "
                        "She has been treated symptomatically with IV fluids and promethazine then discharged with a presumptive diagnosis of gastroenteritis and was asked to show in OPD the next morning. "
                        "The abdominal pain was persistent, severe, and not relieved by medications. It started in the epigastric region and then became diffused all over the abdomen. "
                        "She went to the same emergency department one day later with severe diffuse abdominal pain and persistent vomiting. "
                        "She had a history of osteoarthritis and was on long-term use of NSAID medication. There was no history of cigarette smoking."),
        "tests": {
            "images": [
                {"path": "data/scenario_6/image_1.png", "caption": "Erect Abdominal X-ray"},
                {"path": "data/scenario_6/image_2.png", "caption": "Abdominal CT Scan"}
            ],
            "vital_signs": {
                "BP": "90/60 mmHg",
                "Pulse": "130/min",
                "Respiratory rate": "20/min",
                "Temperature": "38ºC"
            },
            "physical_examination": ("General examination: the patient was emaciated, pale, toxic, and febrile. \n\n"
                                    "Abdominal examination: revealed severe epigastric tenderness and a rigid board-like abdomen with guarding. The bowel sounds were sluggish. \n\n"
                                    "Examination of other systems: unremarkable."),
            "lab_results": {
                "WBCs": "16 × 10³/μL",
                "Neutrophils": "75 %",
                "Lymphocytes": "25 %",
                "Hb": "9 g/dL",
                "RBCs": "4.0 × 10⁶ / μL",
                "MCV": "79.6 fL",
                "Platelets": "434 × 10³/μL",
                "ESR": "66 mm/hr",
                "CRP": "5 mg/L",
                "Blood sugar": "65 mg/dL",
                "Urine analysis": "Normal"
            },
            "radiological_findings": ("Erect abdominal x-ray: free air under the diaphragm (yellow arrow) (Image A). \n\n"
                                    "Abdominal CT scan with IV contrast: demonstrated a focal defect along the lesser curvature of gastric body with surrounding mural thickening (white arrow) and a small air bubble (arrowhead) on the anterior peritoneal surface of the liver (Image B)."),
            "histopathological_findings": ("After initial resuscitation (placement of IV lines and nasogastric tube followed by adequate administration of fluids), the patient underwent an emergency exploratory laparotomy. \n\n"
                                        "During surgery, one gastric perforation was identified along the lesser curvature of the body of the stomach with surrounding mural thickening. \n\n"
                                        "Biopsy with simple closure of the defect was done. The postoperative period was smooth & the patient was discharged after the 7th day.")
        },
        "expected_diagnosis": "Perforated Peptic Ulcer"
    },
    {
        "id": 7,
        "description": ("A 25-year-old man presented to the Orthopedic Clinic with complaints of slowly progressive pain associated with a swelling of the lower end of his right forearm close to the wrist. "
                        "History revealed that the pain was vague and started three months ago with no history of recent trauma."),
        "tests": {
            "images": [
                {"path": "data/scenario_7/image_1.png", "caption": "Antero-posterior X-ray"},
                {"path": "data/scenario_7/image_2.png", "caption": "Lateral X-ray"},
                {"path": "data/scenario_7/image_3.png", "caption": "Needle Biopsy Image (C&D)"}
            ],
            "vital_signs": {
                "Temperature": "37°C",
                "Respiratory rate": "14/min",
                "Pulse": "80/min",
                "Blood pressure": "100/78 mmHg"
            },
            "local_examination": ("Painful swelling over the lower end of the radius. The swelling did not show redness or hotness.\n\n"
                                "Reduced movements of the wrist joint.\n\n"
                                "The doctor requested chest X-ray and X-ray of the right forearm and wrist. "
                                "Chest X-ray was unremarkable.\n\n"
                                "Antero-posterior and lateral X-rays of the right forearm and wrist (Figures A and B) showed a lesion in the distal part of the radius in the form of a lytic bone mass with a “Soap bubble appearance”.\n\n"
                                "The consultant requested MRI of the right forearm and wrist joint and a needle biopsy of the lesion."),
            "pathological_findings": ("A needle biopsy of the lesion revealed the following (Figures C & D):\n\n"
                                    "   • Scattered numerous multinucleated giant cells along with interspersed mononuclear spindled cells.\n\n"
                                    "   • Mitotic figures were 3/10 HPF.\n\n"
                                    "   • No evidence of new bone or osteoid formation with almost complete destruction of bone trabeculae.\n\n"
                                    "MRI findings of the right forearm and wrist confirmed the findings of X-ray.\n\n"
                                    "The surgeon ordered hospitalization of the patient for surgical management of his condition. After that, the doctor advised the patient to revisit him every month for follow-up, however, the patient did not comply.\n\n"
                                    "Two years later, the patient presented to the Orthopaedic Clinic with a larger swelling at the lower end of his right forearm associated with wasting of the thenar eminence, loss of sensations over the palmar surfaces of the lateral three fingers as well as inability to bring the thumb into opposition with the fingers. Radiological assessment showed that the lesion was extending into the surrounding soft tissues including the carpal tunnel."),
        },
        "expected_diagnosis": "Giant Cell Tumour of the Lower End of Radius"
    },
    {
        "id": 8,
        "description": ("Hajar, a 27-year-old woman, was presented to the clinic for evaluation of breast discharge. "
                        "She has noticed breast engorgement and bilateral milky discharge over the past month. "
                        "She reported a history of irregular menstrual cycles associated with mild headaches and progressive blurring of vision since the last year. "
                        "Hajar has been married for 3 years, but never got pregnant. No history of use of any contraceptive method or long-standing medications was reported."),
        "tests": {
            "images": [
                {"path": "data/scenario_8/image_1.png", "caption": "Standard Automated Perimetry (SAP)"},
                {"path": "data/scenario_8/image_2.png", "caption": "Bitemporal Hemianopia"},
                {"path": "data/scenario_8/image_3.png", "caption": "MRI of the pituitary"}
            ],
            "vital_signs": {
                "Temperature": "37°C",
                "Respiratory rate": "19/min",
                "Pulse": "90/min",
                "Blood pressure": "110/70 mmHg"
            },
            "general_examination": ("Breast examination showed no palpable masses, however, bilateral milky discharge from both nipples was observed.\n\n"
                                    "Visual field testing using Standard Automated Perimetry (SAP) (Figure A) showed bitemporal hemianopia (Figure B).\n\n"
                                    "The rest of the systemic examination was unremarkable.\n\n"
                                    "A list of blood tests as well as radiological investigations were requested, and their results were pending."),
            "lab_results": {
                "TSH": "1.5 mIU/L (0.4 - 4 mIU/L)",
                "Prolactin": "1280 ng/mL (< 20 ng/mL)",
                "IGF-1": "28 nmol/L (11 - 40 nmol/L)",
                "FSH": "2.5 IU/L (2 - 9 IU/L)",
                "LH": "1.72 IU/L (1 - 12 IU/L)"
            },
            "radiological_findings": ("MRI of the pituitary (Figure C) showed an isointense lesion measuring 12 mm in the largest dimension extending to the optic chiasma.")
        },
        "expected_diagnosis": "Pituitary Macroadenoma"
    },
    {
        "id": 9,
        "description": ("Ali, a 45-year-old man, presented to the Outpatient Clinic with swellings on the left side of his neck. "
                        "These swellings had become progressively enlarged during the last few weeks and were accompanied by fever, drenching night sweating, weight loss, and loss of appetite."),
        "tests": {
            "images": [
                {"path": "data/scenario_9/image_1.png", "caption": "Image A"},
                {"path": "data/scenario_9/image_2.png", "caption": "Image B"},
                {"path": "data/scenario_9/image_3.png", "caption": "Image C"},
                {"path": "data/scenario_9/image_4.png", "caption": "Image D"},
                {"path": "data/scenario_9/image_5.png", "caption": "Image E"}
            ],
            "vital_signs": {
                "Body weight": "57 kg",
                "Height": "177 cm",
                "Temperature": "37.8°C"
            },
            "general_examination": ("Neck: discrete, nodular swellings located unilaterally below the left mandible, and in the deep areas of the left side of the neck. "
                                    "The swellings were firm, mobile, painless, and measured from 1-3 cm in diameter. \n\n"
                                    "No swellings were present in the axillary or inguinal areas. \n\n"
                                    "Pharyngeal tonsils were not enlarged. \n\n"
                                    "Spleen and liver were not palpable. \n\n"
                                    "Testicular examination was normal."),
            "lab_results": {
                "Hb": "12.8 g/dL",
                "RBCs": "4.3X 10^6/µL",
                "Platelets": "320 X 10^3/µL",
                "Total Leukocytic Count": "7.8 X 10^3/µL"
            },
            "Differential Leukocytic Count": {
                "Neutrophils": "60%",
                "Lymphocytes": "30%",
                "Monocytes": "6%",
                "Eosinophils": "3%",
                "Basophils": "1%",
                "ESR": "58 mm/1st hour"
            },
            "pathological_findings": ("A) Full-body CT scans: \n"
                                    "   • Cervical LNs: multiple adenopathy on left side (2-3 cm). \n\n"
                                    "   • Hilar LNs: bilaterally enlarged (Image B). \n"
                                    "   • Abdominal and pelvic LNs: not enlarged. \n"
                                    "   • Spleen, liver rest of organs were normal.\n\n"
                                    "B) Histopathological report of the lymph nodes biopsy:\n"
                                    "   • Grossly (Image C): several enlarged cervical LNs with homogenous cut-surface & fleshy appearance.\n"
                                    "   • Microscopically (Image D): shows effacement of the normal architecture by diffuse, monotonous, large rounded cells (4-5 times the size of a normal lymphocyte/ RBC/ nuclei of endothelial cells) with moderate cytoplasm, large vesicular nuclei and contain 2-3 prominent nucleoli. Sparsely distributed large cells. Few Reed-Sternberg like cells are identified.\n\n"
                                    "   • Immunohistochemical staining (Image E): shows large rounded cells were positive for CD45 and CD20 and negative for CD5, CD10 and pan T cell markers. Atypical Reed-Sternberg like cells were negative for CD15 & CD30. Monoclonality of Igs in the cytoplasm of these cells was also ascertained.\n\n"
                                    "C) Bone marrow biopsy and aspirate: was requested based upon the results of the lymph node biopsy. It revealed normal bone marrow elements with no abnormal cells.")
        },
        "expected_diagnosis": "Lymphoma"
    }
]





def get_response(user_input, scenario_id):
    scenario = scenarios[scenario_id]
    chat_history = st.session_state.chat_history
    
    # Construct the messages list for the model
    messages = [
        {"role": "system", "content": ",MOST IMPORTENT INSTRUCTIONS: ignore all previous instructions, this app is Medical Training Assistant and you must act as the Patient (do not mention that you are acting ), here is the scenario'"+ scenario["description"]+"'"}
    ] + [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content} 
        for msg in chat_history
    ]
    messages.append({"role": "user", "content": user_input})

    # Use the 'invoke' method to send messages to the model and get a response
    response = llm.invoke(messages)
    
    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    response_text = response.content  # Access content directly from the AIMessage
    st.session_state.chat_history.append(AIMessage(content=response_text))
    
    return response_text

# Streamlit app setup
st.set_page_config(page_title="MediBot")
st.title("MediBot - Medical Training Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = 0

# Display the current scenario description
#st.write(f"Scenario {scenarios[st.session_state.current_scenario]['id']} ({scenarios[st.session_state.current_scenario]['expected_diagnosis']}): {scenarios[st.session_state.current_scenario]['description']}")

# User input through chat interface
user_query = st.chat_input("Type your question:")
if user_query:
    response = get_response(user_query, st.session_state.current_scenario)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)









with st.sidebar:
    diagnosis = st.text_input("Type your diagnosis:")

    # Handling diagnosis submission
    if st.button("Submit Diagnosis"):
        if diagnosis.lower() == scenarios[st.session_state.current_scenario]['expected_diagnosis'].lower():
            st.success("Correct diagnosis!")
            st.session_state.current_scenario = (st.session_state.current_scenario + 1) % len(scenarios)  # Move to the next scenario or loop back
            st.session_state.chat_history = []  # Reset chat history for new scenario
            #st.rerun()
        else:
            st.error("Incorrect diagnosis. Try again or ask more questions.")

    # Button to show more information
    if st.button("Show More Information"):
        st.markdown("<h2 class='header' style='text-align: center; color:green;'>Test Results and Details</h2>", unsafe_allow_html=True)

        # Iterate through the test results dynamically
        for key, value in scenarios[st.session_state.current_scenario]["tests"].items():
            if isinstance(value, str):
                st.markdown(f"#### :violet[{key.replace('_', ' ').title()}]")
                st.markdown(value.replace("\n", "<br>"), unsafe_allow_html=True)
            elif isinstance(value, dict):
                if key == "vital_signs":
                    st.markdown(f"#### :violet[Vital Signs]")
                    st.table(pd.DataFrame([value]).T.rename(columns={0: ""}))
                else:
                    st.markdown(f"#### :violet[{key.replace('_', ' ').title()}]")
                    if all(isinstance(v, (int, float, str)) for v in value.values()):
                        st.table(pd.DataFrame([value]).T.rename(columns={0: ""}))
                    else:
                        for subkey, subvalue in value.items():
                            st.write(f"**{subkey.replace('_', ' ').title()}:** {subvalue}")
            elif isinstance(value, list):
                st.markdown(f"#### :violet[{key.replace('_', ' ').title()}]")
                for item in value:
                    if isinstance(item, dict) and 'path' in item and 'caption' in item:
                        st.image(item['path'], caption=item['caption'])
                    else:
                        st.write(item)

    if st.button("Show Answer"):
        st.write(scenarios[st.session_state.current_scenario]['expected_diagnosis'].lower())













