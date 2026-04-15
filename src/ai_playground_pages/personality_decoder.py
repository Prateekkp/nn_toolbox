import os

import requests
import streamlit as st
from dotenv import load_dotenv

# -------------------------------
# CONFIG (NVIDIA LLM API)
# -------------------------------
load_dotenv()

API_URL = os.getenv("NVIDIA_API_URL") or "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY")
MODEL = os.getenv("NVIDIA_MODEL") or "meta/llama-4-maverick-17b-128e-instruct"


def call_llm(prompt):
    if not API_KEY:
        raise ValueError("Missing API key. Set NVIDIA_API_KEY in your .env file.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_question_rules(mode):
    rules = {
        "Gen Z Vibes": "Topics: situationships, social media, FOMO, late nights, main character moments, red flags.",
        "Daily Life": "Topics: morning routines, habits, food, commute, what drains them, what excites them daily.",
        "Love & Relationships": "Topics: crushes, heartbreaks, situationships, what they avoid, love languages, trust issues.",
        "Ambition & Money": "Topics: goals, money stress, career fears, hustle culture, what they'd do with a crore rupees.",
        "Generic": "Topics: hobbies, opinions, pet peeves, random preferences, would-you-rather style questions."
    }
    return rules.get(mode, rules["Generic"])


def personality_decoder_page():
    PD_PHASE = "pd_phase"
    PD_NAME = "pd_name"
    PD_NAME_LINES = "pd_name_lines"
    PD_MODE = "pd_question_mode"
    PD_AGE = "pd_age"
    PD_PROFESSION = "pd_profession"
    PD_STEP = "pd_dynamic_step"
    PD_ANSWERS = "pd_dynamic_answers"
    PD_CURRENT_Q = "pd_current_question"

    # -------------------------------
    # SESSION STATE INIT
    # -------------------------------
    def reset():
        st.session_state[PD_PHASE] = "intro"       # intro -> dynamic -> result
        st.session_state[PD_NAME] = ""
        st.session_state[PD_NAME_LINES] = []
        st.session_state[PD_MODE] = "Gen Z Vibes"
        st.session_state[PD_AGE] = ""
        st.session_state[PD_PROFESSION] = ""
        st.session_state[PD_STEP] = 0      # 0 to 4 (5 questions)
        st.session_state[PD_ANSWERS] = []
        st.session_state[PD_CURRENT_Q] = ""

    for key, default in {
        PD_PHASE: "intro",
        PD_NAME: "",
        PD_NAME_LINES: [],
        PD_MODE: "Gen Z Vibes",
        PD_AGE: "",
        PD_PROFESSION: "",
        PD_STEP: 0,
        PD_ANSWERS: [],
        PD_CURRENT_Q: ""
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # -------------------------------
    # PHASE 1 - INTRO FORM
    # -------------------------------
    if st.session_state[PD_PHASE] == "intro":
        st.title("Personality Decoder")
        st.write("Three quick things before we get into it.")

        name = st.text_input("What's your name?")
        age = st.text_input("How old are you?")
        profession = st.text_input("What do you do? (student, dev, designer, whatever)")
        question_mode = st.selectbox(
            "What kind of questions do you want?",
            ["Gen Z Vibes", "Daily Life", "Love & Relationships", "Ambition & Money", "Generic"]
        )

        if st.button("Submit"):
            if not name.strip() or not age.strip() or not profession.strip():
                st.warning("Fill everything. All three.")
            else:
                st.session_state[PD_NAME] = name.strip()
                st.session_state[PD_AGE] = age.strip()
                st.session_state[PD_PROFESSION] = profession.strip()
                st.session_state[PD_MODE] = question_mode

                # Pre-generate 5 name astrology lines
                name_prompt = f"""
You are a fun name numerologist.

Given the name "{name}", generate 5 different one-liners about what the name reveals.

Rules:
- Each line should feel different - cover letter energy, number of letters, vowel count, first letter, hidden traits
- Each under 15 words
- Mystical but slightly dramatic
- No emojis, no numbering, no bullet points
- Output exactly 5 lines, one per line, nothing else
"""
                raw = call_llm(name_prompt).strip()
                st.session_state[PD_NAME_LINES] = [line.strip() for line in raw.split("\n") if line.strip()][:5]

                # Generate first dynamic question
                prompt = f"""
You are a sharp, witty personality analyst.

A user just introduced themselves:
- Name: {name}
- Age: {age}
- Profession: {profession}

Generate the FIRST follow-up question to decode their personality deeper.

Rules:
- One line only. Max 10 words. Short and punchy and in plain english.
- {get_question_rules(st.session_state[PD_MODE])}
- Feel like a friend asking, not an interviewer. No emojis.
"""
                st.session_state[PD_CURRENT_Q] = call_llm(prompt).strip()
                st.session_state[PD_PHASE] = "dynamic"
                st.rerun()

    # -------------------------------
    # PHASE 2 - DYNAMIC QUESTIONS
    # -------------------------------
    elif st.session_state[PD_PHASE] == "dynamic":
        name = st.session_state[PD_NAME]
        step = st.session_state[PD_STEP]  # 0-indexed, goes 0 to 4

        st.title("Personality Decoder")

        if st.session_state[PD_NAME_LINES]:
            line = st.session_state[PD_NAME_LINES][min(step, len(st.session_state[PD_NAME_LINES]) - 1)]
            st.warning(f"{name} - {line}")
            st.write("")

        # Progress
        st.caption(f"Question {step + 1} of 5")

        st.subheader(st.session_state[PD_CURRENT_Q])

        user_input = st.text_input("Your answer", key=f"dyn_{step}")

        if st.button("Next"):
            if not user_input.strip():
                st.warning("Don't leave it blank.")
            else:
                st.session_state[PD_ANSWERS].append({
                    "question": st.session_state[PD_CURRENT_Q],
                    "answer": user_input.strip()
                })
                st.session_state[PD_STEP] += 1

                if st.session_state[PD_STEP] < 5:
                    # Generate next question
                    qa_history = "\n".join([
                        f"Q: {qa['question']}\nA: {qa['answer']}"
                        for qa in st.session_state[PD_ANSWERS]
                    ])

                    prompt = f"""
You are a sharp personality analyst.

User profile:
- Name: {st.session_state[PD_NAME]}
- Age: {st.session_state[PD_AGE]}
- Profession: {st.session_state[PD_PROFESSION]}

Their answers so far:
{qa_history}

Generate the NEXT follow-up question.

Rules:
- One line only. Max 10 words. Short and punchy.
- {get_question_rules(st.session_state[PD_MODE])}
- Feel like a friend asking, not an interviewer. No emojis.
"""
                    st.session_state[PD_CURRENT_Q] = call_llm(prompt).strip()
                    st.rerun()

                else:
                    st.session_state[PD_PHASE] = "result"
                    st.rerun()

    # -------------------------------
    # PHASE 3 - RESULT
    # -------------------------------
    elif st.session_state[PD_PHASE] == "result":
        name = st.session_state[PD_NAME]

        qa_history = "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in st.session_state[PD_ANSWERS]
        ])

        final_prompt = f"""
You are a brutally honest, funny Gen Z personality analyst.

User:
- Name: {name}
- Age: {st.session_state[PD_AGE]}
- Profession: {st.session_state[PD_PROFESSION]}

Their answers:
{qa_history}

Decode their personality. Output format (no emojis, no markdown headers, plain text):

Personality Type: [Gen Z label like "Silent Grinder", "Main Character", "Chaotic Neutral", "NPC in someone else's story", etc.]

What you are: [2-3 lines, direct and a little savage]

Your traits:
- [trait 1]
- [trait 2]
- [trait 3]

Honest roast: [1-2 lines, light but real]

Keep the whole thing under 150 words.
"""

        with st.spinner("Decoding..."):
            result = call_llm(final_prompt)

        st.title("Personality Decoder")
        st.write(f"Alright {name}, here it is.")
        st.write("---")
        st.write(result)
        st.write("---")

        if st.button("Run it again"):
            reset()
            st.rerun()
