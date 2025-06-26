# legal_qa_app.py
# Batch legal Q&A with GPT-4.1/4o AND o3 models
import os, re, time, datetime, ssl, certifi, httpx, pandas as pd
from io import BytesIO
from pathlib import Path
import streamlit as st
from openai import OpenAI

# ---------- Helpers ----------------------------------------------
def parse_three_lines(text: str) -> dict:
    out = {"ANSWER": "", "LAW": "", "LINK": ""}
    if not text:
        return out
    for key in out:
        m = re.search(fr"{key}:\s*(.*)", text, re.I)
        out[key] = m.group(1).strip() if m else ""
    return out


LEGAL_QA_SYSTEM_PROMPT = """
You are a legal expert specializing in the government legal framework, laws, regulations, and practices of {economy}. 
Your task is to answer questions related to {economy} legal framework using factual, concise, and up-to-date information. 
Always consult the web search tool for the most current information on {economy}'s laws and regulations before providing an answer.

# Steps:

1. **Read and Understand**: Carefully read and digest the question to fully understand what is being asked.
2. **Consult Sources**: Use the latest official regulatory documents, government publications, and credible sources of {economy}.
3. **Web Search**: Always use the web search tool to ensure information is current and accurate.
4. **Provide Answer**: Think through every step and find the correct legal basis before finalizing your answer.
5. **Format**: Please present your ANSWER as either **yes**, **no**, or **N/A**, followed by the LAW and a clickable LINK to the LAW. Do not include any additional words, commentary, or preamble.

# Output Format:
NO html or Markdown tags in the output. Answer lines should be formatted EXACTLY as follows:
    - ANSWER: Either **yes**, **no**, or **N/A**
    - LAW: <law name, section/article, year>
    - LINK: <official government or legal source URL>

# Example format:
- **Example 1 response:**
  - ANSWER: Yes
  - LAW: Constitution of {economy}, Article XX(X), 2013
  - LINK: https://legislative.gov.in/constitution-of-{economy}

- **Example 2 response:**
  - ANSWER: No
  - LAW: N/A
  - LINK: N/A

# Notes

- Do not provide any unnecessary commentary or speculative information. Always rely on and refer to legal documents.
- Ensure that you find the right answer and legal basis before concluding your response.
- This task requires strict adherence to instructions to ensure clarity and accuracy in legal consultations.
""".strip()


LEGAL_QA_SYSTEM_PROMPT_O = """
You are a legal expert specializing in the government legal framework, laws, regulations, and practices of {economy}. 
Your task is to answer questions related to {economy} legal framework using factual, concise, and up-to-date information. 
Answer yes/no for binary questions. For questions which ask about number of days, time and cost
questions, just provide direct value. No extra commentary. Please make sure you are able to differentiate between a legal 
question (de jure) and a practice question (de facto) and asnwer accordingly. Be very careful and diligent in your task.


# Steps:

1. **Read and Understand**: Carefully read and digest the question to fully understand what is being asked.
2. **Consult Sources**: Use the latest official regulatory documents, government publications, and credible sources of {economy}.
3. **Web Search**: Always use the web search tool to ensure information is current and accurate.
4. **Provide Answer**: Think through every step and find the correct legal basis before finalizing your answer.
5. **Format**: Please present your ANSWER as either **yes**, **no**, or **N/A**, followed by the LAW and a clickable LINK to the LAW. Do not include any additional words, commentary, or preamble.


# Output Format:
NO html or Markdown tags in the output. Answer lines should be formatted EXACTLY as follows:
    - ANSWER: Either **yes**, **no**, or **N/A**
    - LAW: <law name, section/article, year>
    - LINK: <official government or legal source URL>

# Example format:
- **Example 1 response:**
  - ANSWER: Yes
  - LAW: Constitution of {economy}, Article XX(X), 2013
  - LINK: https://legislative.gov.in/constitution-of-{economy}

- **Example 2 response:**
  - ANSWER: No
  - LAW: N/A
  - LINK: N/A

""".strip()



def build_prompt(question: str) -> str:
    # For GPT-4.1/4o we feed only the question in the user field
    return f"Answer this legal question:\n{question}"


# --- universal OpenAI caller
def call_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    system_instructions: str,
    country_code: str,
    temperature: float,
    max_tokens: int,
) -> str:

    # ---------- o-model branch (o3, o3-mini …) ----------
    if model.startswith("o"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_instructions},
                {"role": "user",   "content": prompt}
            ],
            max_output_tokens=max_tokens,   # optional; include only if allowed
            store=True,
            reasoning={
                "effort": "medium", # unchanged
                "summary": "auto" # auto gives you the best available summary (detailed > auto > None)
            }
        )
        return resp.output_text.strip()

    # ---------- GPT-4.1 / 4o branch ----------------------
    resp = client.responses.create(
        model=model,
        instructions=system_instructions,
        input=prompt,
        tools=[{
            "type": "web_search_preview",
            "user_location": {"type": "approximate", "country": country_code},
            "search_context_size": "high",
        }],
        temperature=temperature,
        max_output_tokens=max_tokens,
        top_p=1,
        store=True
    )
    return resp.output_text.strip()



# ---------- Streamlit UI ------------------------------------------------------
st.set_page_config(page_title="B-READY AI Tool", page_icon="⚖️", layout="wide")
st.title("⚖️  B-READY AI Tool")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API key", type="password")
    allowed_models = [
        "gpt-4.1-2025-04-14",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o3",
        "o3-pro",
        "o3-pro-2025-06-10",
        "o3-mini",
        "o4-mini",
        "gpt-4.5-preview"
    ]
    model = st.selectbox("OpenAI model", options=allowed_models)

    is_o_model = model.startswith("o")              # helper flag

    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.1, 0.05,
        disabled=is_o_model,                         # ← greys-out for o-models
        key="temp_slider",
    )
    if is_o_model:
        st.caption("Temperature is ignored for o-models (o3, o3-mini, …).")

    economy = st.text_input("Economy name", value="India")
    country_code = st.text_input("ISO-2 country code", value="IN", max_chars=2)
    st.markdown("---")
    st.subheader("Advanced")
    max_retry   = st.number_input("Max API retries", 0, 5, 2)
    #temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens  = st.number_input("Max output tokens", 50, 50000, 1500, 50)
    sleep_time  = st.slider("Sleep between retries (sec)", 0.0, 5.0, 0.7, 0.1)
    http_timeout= st.number_input("HTTP timeout (sec)", 10, 600, 600, 5)
    extra_ca    = st.text_input("Extra CA bundle (optional)")

uploaded_file = st.file_uploader(
    "Upload Excel with **Reference** and **Question** columns", type=["xlsx"]
)

if uploaded_file:
    df_questions = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded file:")
    st.dataframe(df_questions.head())

    if st.button("Run Q&A"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
            st.stop()

        ctx = ssl.create_default_context(cafile=certifi.where())
        if extra_ca:
            ctx.load_verify_locations(cafile=extra_ca)
        http_client = httpx.Client(
            verify=ctx, timeout=http_timeout, follow_redirects=True
        )
        client = OpenAI(api_key=api_key, http_client=http_client)

        progress_bar   = st.progress(0, text="Starting …")
        status_text    = st.empty()
        live_table_ph  = st.empty()

        df_answers = df_questions.copy()
        for col in ("Answer", "Legal Basis", "Legal Link"):
            if col not in df_answers.columns:
                df_answers[col] = ""

        total = len(df_answers)

        title_ph   = st.empty()
        answer_ph  = st.empty()
        for idx, row in df_answers.iterrows():
            q_ref, question = row["Reference"], row["Question"]
            status_text.markdown(f"**Processing:** `{q_ref}`")

            attempt = 0
            while attempt <= max_retry:
                try:
                    prompt = build_prompt(question)
                    sys_instr = LEGAL_QA_SYSTEM_PROMPT.format(economy=economy)
                    raw = call_openai(
                        client,
                        model,
                        prompt,
                        sys_instr,
                        country_code.upper(),
                        temperature,
                        max_tokens,
                    )

                    title_ph.markdown(f"### Response for **{q_ref}**")
                #    st.markdown(f"### Response for **{q_ref}**")
                #    st.code(raw, language="text")
                    answer_ph.code(raw, language="text")
                    parsed = parse_three_lines(raw)
                    df_answers.at[idx, "Answer"]      = parsed["ANSWER"]
                    df_answers.at[idx, "Legal Basis"] = parsed["LAW"]
                    df_answers.at[idx, "Legal Link"]  = parsed["LINK"]
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > max_retry:
                        df_answers.at[idx, "Answer"] = f"API error: {e}"
                        st.error(f"[{q_ref}] {e}")
                        break
                    time.sleep(sleep_time)

            live_table_ph.dataframe(df_answers.iloc[: idx + 1], use_container_width=True)
            progress_bar.progress(int((idx + 1) / total * 100))

        progress_bar.empty()
        status_text.success("Completed!")
        st.subheader("Full results")
        st.dataframe(df_answers, use_container_width=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"{Path(uploaded_file.name).stem}_{economy}_{model}_{timestamp}.xlsx"
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as wr:
            df_answers.to_excel(wr, index=False)
        buffer.seek(0)
        st.download_button(
            "Download results as Excel", buffer, file_name=out_name
        )
else:
    st.info("Upload an Excel file to get started.")
