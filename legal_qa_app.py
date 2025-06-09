# legal_qa_app.py
# Batch legal Q&A with GPT-4.1/4o AND o3 models
import os, re, time, datetime, ssl, certifi, httpx, pandas as pd
from io import BytesIO
from pathlib import Path
import streamlit as st
from openai import OpenAI

# ---------- Helpers --------------------------------------------------------
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

You will be tasked to answer questions related to the government legal framework.

Your thinking should be thorough and so it's fine if it's very long. You can think step by step before and after each action you decide to take.

Your job is to answer legal questions factually, concisely, and using your knowledge of current, official legal sources,

as well as consulting the web_search. Always concult web search tool for most upto data information before you answer please.

NEVER end your turn without having finding the right answer to the question.

Take your time and think through every step. For legal basis, please ensure you get **legal clauses**, chapter numbers, section numbers, article numbers, etc.

Follow these rules:
- First, read the question carefully and digest it. Deeply Understand the question.
- You must answer using the latest official regulatory documents, government publications, and credible sources.
- Output MUST be formatted in three lines exactly:
  ANSWER: <direct answer>
  LAW: <law name, section/article, year>
  LINK: <official government or legal source URL>
- If the ANSWER is yes, always return the legal basis and the correct link to the legal basis.
- Reflect and validate comprehensively.
- For binary questions, only answer yes or no. IF a law or regulation is not applicable, return 'NA'.


NO commentary, NO extra lines, NO preamble, NO citations beyond those required above.
PLEASE do NOT speculate. DO not halucinate. If the answer is not in your training, search internet for the correct answer.

Example:
ANSWER: Yes
LAW: Constitution of India, Article 21, 1950
LINK: https://legislative.gov.in/constitution-of-india

Example:
ANSWER: No
LAW: N/A
LINK: N/A
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
    if model.startswith("o3"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_instructions},
                {"role": "user",   "content": prompt}
            ],
            max_output_tokens=max_tokens,   # optional; include only if allowed
            store=False,
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
        store=False,
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
        "o3-mini",
        "o4-mini",
        "o4-mini",
        "gpt-4.5-preview"
    ]
    model = st.selectbox("OpenAI model", options=allowed_models)

    is_o_model = model.startswith("o3")              # helper flag

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
    max_tokens  = st.number_input("Max output tokens", 50, 2000, 512, 50)
    sleep_time  = st.slider("Sleep between retries (sec)", 0.0, 5.0, 0.7, 0.1)
    http_timeout= st.number_input("HTTP timeout (sec)", 10, 120, 30, 5)
    extra_ca    = st.text_input("Extra CA bundle (optional)")

uploaded_file = st.file_uploader(
    "Upload Excel with **Reference** and **Question** columns", type=["xlsx"]
)

if uploaded_file:
    df_questions = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded file:")
    st.dataframe(df_questions.head())

    if st.button("▶️  Run Q&A"):
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
                    st.markdown(f"### Response for **{q_ref}**")
                    st.code(raw, language="text")
                    parsed = parse_three_lines(raw)
                    df_answers.at[idx, "Answer"]      = parsed["ANSWER"]
                    df_answers.at[idx, "Legal Basis"] = parsed["LAW"]
                    df_answers.at[idx, "Legal Link"]  = parsed["LINK"]
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > max_retry:
                        df_answers.at[idx, "Answer"] = f"API error: {e}"
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
            "💾 Download results as Excel", buffer, file_name=out_name
        )
else:
    st.info("⬆️  Upload an Excel file to get started.")
