# Task Description: Hospital ChatBot
You are an AI engineer at WVU Medicine Health System. Your CIO wants to build a system so that employees and patients can ask questions related to any document uploaded on the hospital system website, [wvumedicine.org](https://wvumedicine.org). Your team has already downloaded and indexed the document set, and as the AI engineer you are tasked with building the pipeline to support any queries against the pdfs on the website.

The CIO fielded questions, desired answers, and source documents from doctors, patients, and the legal team. You need to make sure that your pipeline answers these questions correctly with the specified information.

Here's some other quick information about the document set:
- Documents range from are 1-50 pages long
- There are 220 pdfs in the document set
- Some duplicate documents exist


| Question | Answer | Source Document |
| :-------- | :------ | :--------------- |
| What is the over-award policy?  | <pre>An over-award situation would exist in the event that a student’s financial aid package exceeds the student’s financial need.  Adjustments to the student’s financial aid package will be made accordingly in over-award situations. 1. Pell grant awards will not be adjusted in recalculating the student’s financial aid unless an overpayment occurs (See Section III - Overpayments) 2. Direct Loan awards (Subsidized, Unsubsidized, & PLUS) will be adjusted by one of the following: a. Canceling or reducing the award prior to the 1st disbursement. b. Canceling or reducing subsequent loan disbursements. c. Replacing the EFC by converting Subsidized into Unsubsidized loan amounts. 3. Partial disbursements will not be made.  If necessary, WVUH will return the entire award and recalculate the student’s loan eligibility.  A new, corrected disbursement will be issued. </pre> | [Click to open source document](/task/documents/policy_pdfs/1.011-Return-of-Title-IV-funds.pdf)
| Who is on the investigational drug services team?  | <pre> The Investigational Drug Services (IDS) team is comprised of pharmacists and pharmacy technicians. </pre>| [Click to open source document](/task/documents/policy_pdfs/wvu_ids_pharmacy_sop02_investigational_drug_pharmacy_staff_training_checklist-132024.pdf)
| What languages can we translate in the hospital?  | <pre>Spanish, Chinese, French, German, Arabic, Vietnamese, Korean, Japanese, Tagalog, Italian, Thai, Nepali, Persian, Russian, Urdu, American Sign Language </pre> | [Click to open source document](/task/documents/policy_pdfs/Request-a-translator.pdf)
| Can we translate Spanish?  | <pre>Yes</pre> | [Click to open source document](/task/documents/policy_pdfs/Request-a-translator.pdf)
| What is the safety policy for job shadowing at Berkeley Medical Center? | <pre> Job-shadowers must stay with their host at all times and cannot wander off. Job shadowing is a strictly hands off observation, and the shadower must follow all instructions from their host. The host will guide the job-shadower in regards to infection control, safety and hazardous materials to ensure their safety. </pre> | [Click to open source document](/task/documents/policy_pdfs/Job-Shadow-Program-overview.pdf)
| How many beds are in Berkeley Medical Center?  | <pre> There are 195 beds in Berkeley Medical Center. </pre> | [Click to open source document](/task/documents/policy_pdfs/bmc_jmc_community_health_needs_assessment_2022.pdf)
| How many beds are in Jefferson Medical Center?  | <pre> There are 45 beds in Jefferson Medical Center. </pre> | [Click to open source document](/task/documents/policy_pdfs/bmc_jmc_community_health_needs_assessment_2022.pdf)
| What are the different admissions policies? | <pre>There are three admissions policies: **DMS Admissions (Policy No. 2.027)** - DMS program is non-discriminatory, requires an associate degree in Allied Health, and an objective screening mechanism is used. **Echocardiography Admissions Policy** is for the echocardiography program is  non-discriminatory, and requires at least an associates degree or higher.  **MRI Admissions (Policy No. 2.001)** - MRI program is non-discriminatory, requires Radiologic Technology graduation, ARRT certification, and an associate's degree or higher. </pre> | Multiple Source Documents: [Click here.](/task/documents/policy_pdfs/DMS-Admissions-Policy.pdf) [Click here.](/task/documents/policy_pdfs/2.006-Admissions-23.pdf) [Click here.](/task/documents/policy_pdfs/Echocardiography-Admissions-Policy.pdf)


## Stage 2: Supporting Erroneous Inputs
After initial deployment your team has received feedback that the pipeline is not working. Upon log analysis, it seems that some users are trying to use it like Google Search with keywords or have many typos in their question.

Evolve your pipeline to support erroneous query types. Some examples are included below.

| Question | Erroneous query examples |
| :-------- |  :--------------- |
| What is the over-award policy | over-award policy, over award |
| What languages can we translate in the hospital? | languages we translate | 
| How many beds are in Berkeley Medical Center? | num beds berkeley |
| How many beds are in Jefferson Medical Center? | jefferson num beds |

## Stage 3: Conversational Ready
Evolve your pipeline so that if the user asks an irrelevant question, the model outputs:
"I am a chatbot for answering questions about WVU Medicine's health system. Your question does not seem relevant, but feel free to ask me about information on our website."


# [Document Set](/task/documents/)
Your documents are pre-indexed into vector stores. Here's a table of the valid combinations of chunk sizes and overlaps you can use with your retriever.

| Chunk* | 0   | 10  | 25  | 50  | 100 | 200 | 400 |
|:----------------------------|-----|-----|-----|-----|-----|-----|-----|
| 100                        |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |     |     |
| 200                        |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |     |
| 400                        |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| 500                        |     |     |     |     |  ✅  |     |     |
| 800                        |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| 1000                       |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| 1500                       |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |     |
| 2000                       |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
*size (down), overlap Across

# [Starter Code](/task/pipeline.py)

# Other Helpful Resources
- [Code Onboarding](/ONBOARDING.md)
- [Coding/LLM Cheat Sheet](/CHEAT_SHEET.md)