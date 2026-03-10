"""
EXCELA - Excellence & Competency for Learning, Achievement & Talent
PCU Career & Employability Center (CEC)
"""

import os
import re
import time
import requests
from generate_report import generate_report
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR     = "./chroma_excela"
INTERVIEW_TIME = 15 * 60
MODEL          = "excela:latest"
TEMPERATURE    = 0.65

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
}

# ─── SYSTEM PROMPTS ────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "national": """You are a senior HR interviewer at a top Indonesian company.
You are conducting a REAL job interview with a fresh graduate. Do NOT act as a coach or mentor during the interview.

The interview has TWO parts — cover both naturally:

PART 1 — HR & Logistics (cover these through conversation):
- Self introduction
- Motivation: why this company, why this role
- Salary expectation (IDR/month) — probe if unrealistic
- Availability / start date
- Willingness to relocate across Indonesia
- Work type preference: WFO / hybrid / WFH
- Benefits needs (BPJS, THR, transport, meal allowance)
- Career goals in 3-5 years

PART 2 — Domain Knowledge (role-specific, use the job market context):
- Ask at least 3 technical or domain-specific questions relevant to the role
- Use real industry terminology, scenarios, and case questions
- Probe for depth — push back on shallow answers
- Assess whether the candidate truly understands the field

Rules:
- Ask ONE question at a time
- React naturally ("Interesting, bisa dijelaskan lebih lanjut?")
- Mix Bahasa Indonesia and English as Indonesian corporates do
- Do NOT re-introduce yourself after opening
- Do NOT reveal you are an AI
- Close professionally: "Kami akan menghubungi Anda dalam 1-2 minggu ke depan" """,

    "international": """You are a senior HR recruiter at a Fortune 500 multinational company.
You are conducting a REAL job interview with a fresh graduate. Do NOT act as a coach or mentor during the interview.

The interview has TWO parts — cover both naturally:

PART 1 — HR & Logistics (cover these through conversation):
- 90-second elevator pitch / self introduction
- Motivation: why this company specifically, not competitors
- Expected total compensation (USD, annual)
- Notice period / earliest start date
- Willingness to relocate or travel internationally
- Work type: office / hybrid / remote / time zone flexibility
- Visa or sponsorship requirements
- 5-year global career goals

PART 2 — Domain Knowledge (role-specific, use the job market context):
- Ask at least 3 technical or domain-specific questions relevant to the role
- Use STAR method for behavioral questions
- Push for metrics, data, and specific outcomes
- Challenge assumptions — Fortune 500 bar is high
- Assess whether the candidate truly understands the field

Rules:
- Ask ONE question at a time
- English only — professional business tone
- React naturally ("That's interesting, can you elaborate?")
- Do NOT re-introduce yourself after opening
- Do NOT reveal you are an AI
- Close professionally: "Our team will be in touch within 2 weeks" """
}


# ─── SCRAPERS ──────────────────────────────────────────────────────────────────
def scrape_wikipedia(career_goal: str) -> list:
    docs = []
    try:
        url  = f"https://en.wikipedia.org/wiki/{career_goal.replace(' ', '_')}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        content = soup.find("div", {"id": "mw-content-text"})
        if content:
            text = " ".join(
                p.get_text() for p in content.find_all("p")[:10]
                if len(p.get_text()) > 50
            )
            if text:
                docs.append(Document(
                    page_content=f"Industry overview — {career_goal}:\n{text}",
                    metadata={"source": "Wikipedia"}
                ))
                print(f"    ✓ Wikipedia: {len(text)} chars")
    except Exception as e:
        print(f"    ✗ Wikipedia: {e}")
    return docs


def scrape_google(career_goal: str, company_type: str) -> list:
    docs = []
    location = "Indonesia" if company_type == "national" else "global Fortune 500"
    queries = [
        f"{career_goal} job description responsibilities 2024",
        f"{career_goal} technical skills requirements {location}",
        f"{career_goal} salary fresh graduate {location} 2024",
        f"{career_goal} interview questions domain specific",
    ]
    collected = 0
    for q in queries:
        try:
            url  = f"https://www.google.com/search?q={q.replace(' ', '+')}&hl=en"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = " ".join(s.get_text() for s in soup.find_all("div", class_="VwiC3b")[:6])
            if text:
                docs.append(Document(
                    page_content=f"'{q}':\n{text}",
                    metadata={"source": "Google", "query": q}
                ))
                collected += 1
        except Exception as e:
            pass
    if collected:
        print(f"    ✓ Google: {collected} results")
    return docs


def scrape_linkedin_jobs(career_goal: str, company_type: str) -> list:
    docs = []
    try:
        query    = career_goal.replace(" ", "%20")
        location = "Indonesia" if company_type == "national" else "Worldwide"
        url      = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={query}&location={location}&start=0"
        resp     = requests.get(url, headers=HEADERS, timeout=15)
        soup     = BeautifulSoup(resp.text, "html.parser")
        jobs     = soup.find_all("li")[:8]

        texts = []
        for job in jobs:
            title   = job.find("h3", class_="base-search-card__title")
            company = job.find("h4", class_="base-search-card__subtitle")
            loc     = job.find("span", class_="job-search-card__location")
            if title:
                parts = [title.get_text(strip=True)]
                if company: parts.append(company.get_text(strip=True))
                if loc:     parts.append(loc.get_text(strip=True))
                texts.append(" | ".join(parts))

        if texts:
            docs.append(Document(
                page_content=f"LinkedIn job listings — {career_goal}:\n" + "\n".join(texts),
                metadata={"source": "LinkedIn Jobs"}
            ))
            print(f"    ✓ LinkedIn: {len(texts)} listings")
        else:
            print(f"    ~ LinkedIn: blocked, skipping")
    except Exception as e:
        print(f"    ✗ LinkedIn: {e}")
    return docs


def scrape_jobstreet(career_goal: str) -> list:
    docs = []
    try:
        query = career_goal.replace(" ", "-").lower()
        url   = f"https://www.jobstreet.co.id/en/job-search/{query}-jobs/"
        resp  = requests.get(url, headers=HEADERS, timeout=12)
        soup  = BeautifulSoup(resp.text, "html.parser")
        cards = soup.find_all("article")[:8]

        texts = []
        for card in cards:
            title   = card.find("a", {"data-automation": "jobTitle"})
            company = card.find("span", {"data-automation": "jobCardCompany"})
            salary  = card.find("span", {"data-automation": "jobSalary"})
            parts   = []
            if title:   parts.append(title.get_text(strip=True))
            if company: parts.append(company.get_text(strip=True))
            if salary:  parts.append(f"Salary: {salary.get_text(strip=True)}")
            if parts:   texts.append(" | ".join(parts))

        if texts:
            docs.append(Document(
                page_content=f"Jobstreet listings — {career_goal}:\n" + "\n".join(texts),
                metadata={"source": "Jobstreet"}
            ))
            print(f"    ✓ Jobstreet: {len(texts)} listings")
        else:
            print(f"    ~ Jobstreet: no listings parsed")
    except Exception as e:
        print(f"    ✗ Jobstreet: {e}")
    return docs


def scrape_indeed(career_goal: str, company_type: str) -> list:
    docs = []
    try:
        query    = career_goal.replace(" ", "+")
        location = "Indonesia" if company_type == "national" else ""
        url      = f"https://www.indeed.com/jobs?q={query}&l={location}&explvl=entry_level"
        resp     = requests.get(url, headers=HEADERS, timeout=12)
        soup     = BeautifulSoup(resp.text, "html.parser")
        cards    = soup.find_all("td", class_="resultContent")[:8]

        texts = []
        for card in cards:
            title   = card.find("h2", class_="jobTitle")
            company = card.find("span", class_="companyName")
            salary  = card.find("div", class_="salary-snippet-container")
            parts   = []
            if title:   parts.append(title.get_text(strip=True))
            if company: parts.append(company.get_text(strip=True))
            if salary:  parts.append(f"Salary: {salary.get_text(strip=True)}")
            if parts:   texts.append(" | ".join(parts))

        if texts:
            docs.append(Document(
                page_content=f"Indeed listings — {career_goal}:\n" + "\n".join(texts),
                metadata={"source": "Indeed"}
            ))
            print(f"    ✓ Indeed: {len(texts)} listings")
        else:
            print(f"    ~ Indeed: no listings parsed")
    except Exception as e:
        print(f"    ✗ Indeed: {e}")
    return docs


def build_hr_context(career_goal: str, company_type: str) -> Document:
    if company_type == "national":
        text = f"""
HR Context — {career_goal} fresh graduate at Indonesian companies:

Salary benchmark (IDR/month, 2024):
- Entry level: Rp 4.000.000 – Rp 8.000.000
- UMR Jakarta 2024: Rp 5.067.381
- Benefits: BPJS Kesehatan, BPJS Ketenagakerjaan, THR (1x gaji), transport/meal allowance
- Probation: 3 months standard

Domain knowledge areas to test for {career_goal}:
- Core technical concepts specific to {career_goal}
- Industry tools, software, and methodologies used in {career_goal}
- Real workplace scenarios and problem-solving in {career_goal}
- Indonesian regulatory and compliance context for {career_goal}
- Case questions relevant to {career_goal} daily responsibilities
"""
    else:
        text = f"""
HR Context — {career_goal} fresh graduate at Fortune 500:

Compensation benchmark (USD/year, 2024):
- Entry level: $50,000 – $120,000 depending on company/city
- Total comp: base + signing bonus + RSU + health + 401k
- Relocation package often available

Domain knowledge areas to test for {career_goal}:
- Advanced technical concepts specific to {career_goal}
- Global industry tools, frameworks, and best practices for {career_goal}
- Data-driven problem solving scenarios in {career_goal}
- Cross-functional and global team collaboration in {career_goal}
- STAR behavioral questions tied to {career_goal} competencies
"""
    return Document(page_content=text, metadata={"source": "EXCELA HR Framework"})


def scrape_job_data(career_goal: str, company_type: str) -> list:
    print(f"\n  Gathering market data for: {career_goal}")
    all_docs = []
    print("  [1/5] Wikipedia...")
    all_docs.extend(scrape_wikipedia(career_goal))
    print("  [2/5] Google...")
    all_docs.extend(scrape_google(career_goal, company_type))
    print("  [3/5] LinkedIn...")
    all_docs.extend(scrape_linkedin_jobs(career_goal, company_type))
    print("  [4/5] Jobstreet...")
    all_docs.extend(scrape_jobstreet(career_goal))
    print("  [5/5] Indeed...")
    all_docs.extend(scrape_indeed(career_goal, company_type))
    all_docs.append(build_hr_context(career_goal, company_type))
    print(f"\n  {len(all_docs)} sources loaded")
    return all_docs


# ─── RAG ───────────────────────────────────────────────────────────────────────
def build_rag(docs: list) -> Chroma:
    print("\n  Building knowledge base...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80
    ).split_documents(docs)
    print(f"  {len(chunks)} chunks indexed")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        persist_directory=CHROMA_DIR
    )
    print("  Knowledge base ready\n")
    return vectordb


# ─── OUTCOME DECISION ──────────────────────────────────────────────────────────
def evaluate_outcome(llm, interview_log: str, student_name: str,
                     career_goal: str, company_label: str) -> dict:
    """Ask EXCELA to score and decide hiring outcome."""

    eval_messages = [
        SystemMessage(content="""You are a strict hiring committee evaluating a job interview transcript.
Your job is to score the candidate and decide their hiring outcome.

Scoring criteria (each out of 20):
- Communication & Clarity       : how well they expressed themselves
- Domain Knowledge              : depth of role-specific technical knowledge
- HR Fit (salary/availability)  : realistic expectations, flexibility
- Motivation & Culture Fit      : genuine interest, alignment with company values
- Confidence & Professionalism  : composure, structure, maturity

After scoring, decide outcome:
- HIRED       : total score >= 80 — strong across all areas
- ON HOLD     : total score 60–79 — promising but needs verification or one more round
- REJECTED    : total score < 60  — does not meet the bar

Respond ONLY in this exact format:
SCORE_COMMUNICATION: X
SCORE_DOMAIN: X
SCORE_HR_FIT: X
SCORE_MOTIVATION: X
SCORE_CONFIDENCE: X
TOTAL: XX
OUTCOME: HIRED|ON_HOLD|REJECTED
REASON: (2-3 sentences explaining the decision honestly)
FEEDBACK: (3 specific improvement points for the candidate)"""),
        HumanMessage(content=(
            f"Candidate: {student_name}\n"
            f"Role: {career_goal}\n"
            f"Company type: {company_label}\n\n"
            f"Interview transcript:\n{interview_log}\n\n"
            f"Evaluate and decide."
        ))
    ]

    response = llm.invoke(eval_messages)
    raw      = response.content

    # Parse scores
    def extract(key):
        m = re.search(rf"{key}:\s*(\d+)", raw)
        return int(m.group(1)) if m else 0

    scores = {
        "communication" : extract("SCORE_COMMUNICATION"),
        "domain"        : extract("SCORE_DOMAIN"),
        "hr_fit"        : extract("SCORE_HR_FIT"),
        "motivation"    : extract("SCORE_MOTIVATION"),
        "confidence"    : extract("SCORE_CONFIDENCE"),
        "total"         : extract("TOTAL"),
    }

    outcome_match = re.search(r"OUTCOME:\s*(HIRED|ON_HOLD|REJECTED)", raw)
    outcome       = outcome_match.group(1) if outcome_match else "ON_HOLD"

    reason_match  = re.search(r"REASON:\s*(.+?)(?=FEEDBACK:|$)", raw, re.DOTALL)
    reason        = reason_match.group(1).strip() if reason_match else ""

    feedback_match = re.search(r"FEEDBACK:\s*(.+)", raw, re.DOTALL)
    feedback       = feedback_match.group(1).strip() if feedback_match else ""

    return {
        "scores"   : scores,
        "outcome"  : outcome,
        "reason"   : reason,
        "feedback" : feedback,
    }


def print_outcome(result: dict, student_name: str, career_goal: str,
                  company_label: str, company_type: str):
    """Print the HR decision letter."""

    scores  = result["scores"]
    outcome = result["outcome"]

    # Score table
    print("\n" + "═" * 58)
    print("  EXCELA INTERVIEW EVALUATION")
    print(f"  {student_name} | {career_goal} | {company_label}")
    print("═" * 58)
    print(f"  Communication & Clarity   : {scores['communication']:>2}/20")
    print(f"  Domain Knowledge          : {scores['domain']:>2}/20")
    print(f"  HR Fit (salary/avail.)    : {scores['hr_fit']:>2}/20")
    print(f"  Motivation & Culture Fit  : {scores['motivation']:>2}/20")
    print(f"  Confidence & Professional : {scores['confidence']:>2}/20")
    print("  " + "─" * 40)
    print(f"  TOTAL SCORE               : {scores['total']:>2}/100")
    print("═" * 58)

    # Outcome banner
    if outcome == "HIRED":
        print("""
  ╔══════════════════════════════════════════════╗
  ║  ✅  CONGRATULATIONS — YOU ARE HIRED!        ║
  ╚══════════════════════════════════════════════╝""")
        if company_type == "national":
            print(f"""
  Dear {student_name},

  Selamat! Kami dengan bangga memberitahukan bahwa Anda
  telah berhasil melewati tahap interview untuk posisi
  {career_goal} di perusahaan kami.

  Tim HR kami akan menghubungi Anda dalam 3-5 hari kerja
  untuk membahas detail penawaran kerja, termasuk:
  - Surat penawaran (offer letter)
  - Tanggal mulai bekerja
  - Paket kompensasi final

  Selamat bergabung!
  — HR Department""")
        else:
            print(f"""
  Dear {student_name},

  Congratulations! We are pleased to inform you that you
  have successfully passed the interview for the position
  of {career_goal}.

  Our HR team will contact you within 3–5 business days
  to discuss:
  - Formal offer letter
  - Compensation package details
  - Onboarding schedule

  Welcome to the team!
  — HR Department""")

    elif outcome == "ON_HOLD":
        print("""
  ╔══════════════════════════════════════════════╗
  ║  🔄  STATUS: ON HOLD — SECOND ROUND         ║
  ╚══════════════════════════════════════════════╝""")
        if company_type == "national":
            print(f"""
  Dear {student_name},

  Terima kasih telah meluangkan waktu untuk interview
  posisi {career_goal} bersama kami.

  Kami sangat menghargai antusiasme dan potensi Anda.
  Saat ini kami masih dalam proses evaluasi kandidat
  dan akan menghubungi Anda dalam 1-2 minggu ke depan
  mengenai tahapan selanjutnya.

  Mohon untuk tetap standby dan pastikan nomor kontak
  Anda aktif.

  Terima kasih,
  — HR Department""")
        else:
            print(f"""
  Dear {student_name},

  Thank you for your time interviewing for the
  {career_goal} position with us.

  We were impressed by several aspects of your profile.
  We are currently finalizing our candidate evaluations
  and will reach out within 1–2 weeks regarding
  next steps, which may include a second interview.

  Please keep your contact details updated.

  Best regards,
  — HR Department""")

    else:  # REJECTED
        print("""
  ╔══════════════════════════════════════════════╗
  ║  ❌  STATUS: NOT SELECTED AT THIS TIME       ║
  ╚══════════════════════════════════════════════╝""")
        if company_type == "national":
            print(f"""
  Dear {student_name},

  Terima kasih telah melamar dan mengikuti proses
  interview untuk posisi {career_goal} di perusahaan kami.

  Setelah melalui proses evaluasi yang cermat, kami
  menyampaikan bahwa kami tidak dapat melanjutkan
  aplikasi Anda ke tahap berikutnya saat ini.

  Keputusan ini bukan berarti Anda tidak berkompeten —
  kami mendorong Anda untuk terus berkembang dan
  mencoba kembali di masa mendatang.

  Terima kasih,
  — HR Department""")
        else:
            print(f"""
  Dear {student_name},

  Thank you for your interest in the {career_goal}
  position and for the time you invested in our
  interview process.

  After careful consideration, we have decided to
  move forward with other candidates whose experience
  more closely matches our current requirements.

  We encourage you to continue developing your skills
  and welcome you to apply again in the future.

  Best regards,
  — HR Department""")

    # EXCELA coaching section
    print("\n" + "═" * 58)
    print("  EXCELA PRIVATE COACHING NOTES")
    print("  (This section is for your personal development)")
    print("═" * 58)
    print(f"\n  Decision rationale:\n  {result['reason']}\n")
    print(f"  Areas to work on:\n  {result['feedback']}\n")
    print("═" * 58)
    print(f"  Keep growing, {student_name}! — EXCELA | PCU CEC")
    print("═" * 58)


# ─── INTERVIEW ─────────────────────────────────────────────────────────────────
def run_interview(student_name: str, career_goal: str,
                  company_type: str, vectordb: Chroma):

    llm       = ChatOllama(model=MODEL, temperature=TEMPERATURE)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    context_docs = retriever.invoke(
        f"{career_goal} domain knowledge skills interview salary requirements"
    )
    context = "\n\n".join([
        f"[{d.metadata.get('source','Source')}]\n{d.page_content}"
        for d in context_docs
    ])

    company_label = "Top Indonesian Company" if company_type == "national" else "Fortune 500 Company"

    history = [
        SystemMessage(content=(
            f"{SYSTEM_PROMPTS[company_type]}\n\n"
            f"Position: {career_goal}\n"
            f"Candidate: {student_name} (fresh graduate, Petra Christian University, Surabaya)\n\n"
            f"Real Job Market & Domain Context:\n{context}"
        ))
    ]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           EXCELA - PCU Career & Employability Center     ║
╠══════════════════════════════════════════════════════════╣
║  Candidate : {student_name:<41}║
║  Role      : {career_goal:<41}║
║  Target    : {company_label:<41}║
║  Duration  : 15 minutes                                  ║
╠══════════════════════════════════════════════════════════╣
║  Type 'exit' to end session early                        ║
╚══════════════════════════════════════════════════════════╝
""")

    start_time = time.time()
    qa_count   = 0

    # Opening
    history.append(HumanMessage(
        content=f"Start the interview. Greet {student_name} and begin."
    ))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))
    print(f"\nHR: {response.content}\n")
    qa_count += 1

    # Loop
    while True:
        elapsed   = time.time() - start_time
        remaining = INTERVIEW_TIME - elapsed

        if remaining <= 0:
            close_note = (
                "[INTERNAL: Time is up. Close the interview professionally — "
                "thank the candidate, tell them HR will follow up within 1-2 weeks, "
                "and say goodbye. Do NOT ask any more questions.]"
            )
            history.append(HumanMessage(content=close_note))
            response = llm.invoke(history)
            history.append(AIMessage(content=response.content))
            print(f"\nHR: {response.content}\n")
            break

        mins = int(remaining // 60)
        secs = int(remaining % 60)

        user_input = input(f"[{mins:02d}:{secs:02d}] {student_name}: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "selesai"]:
            close_note = (
                "[INTERNAL: The candidate has ended the session. "
                "Close professionally — thank them, mention HR will follow up, say goodbye.]"
            )
            history.append(HumanMessage(content=close_note))
            response = llm.invoke(history)
            history.append(AIMessage(content=response.content))
            print(f"\nHR: {response.content}\n")
            break

        if mins < 2:
            note = (
                f"[INTERNAL: Only {mins}m {secs}s left. "
                f"Finish current topic and begin closing. Do not start new questions.]"
            )
        else:
            note = (
                f"[INTERNAL: {qa_count} exchanges, {mins}m {secs}s remaining. "
                f"Continue naturally. Cover domain-specific questions if not yet asked.]"
            )

        history.append(HumanMessage(content=f"{user_input}\n\n{note}"))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))
        print(f"\nHR: {response.content}\n")
        qa_count += 1

        # Check if HR naturally closed the interview via keyword signals
        closing_signals = [
            # Indonesian
            "terima kasih atas waktu", "kami akan menghubungi", "sampai jumpa",
            "selamat tinggal", "semoga berhasil", "kami akan follow up",
            "dalam 1-2 minggu", "dalam 3-5 hari", "sampai bertemu",
            # English
            "thank you for your time", "we will be in touch", "we'll be in touch",
            "we will get back to you", "we'll get back to you",
            "good luck", "goodbye", "best of luck", "have a great day",
            "that concludes our interview", "that wraps up", "interview is now complete",
            "we will contact you", "our team will reach out",
            "within 1-2 weeks", "within 3-5 business days",
        ]
        if any(signal in response.content.lower() for signal in closing_signals):
            print("\n  [HR has concluded the interview]\n")
            break

    # Waiting simulation
    print("\n" + "=" * 58)
    if company_type == "national":
        print("  Menunggu keputusan HR...")
        print("  Tim HR sedang mengevaluasi hasil interview Anda.")
    else:
        print("  Awaiting HR decision...")
        print("  The hiring committee is reviewing your interview.")
    print("=" * 58)
    for i in range(3):
        time.sleep(1)
        print(f"  {'.' * (i+1)}")
    print("\n  You have received a message from HR.\n")
    time.sleep(1)

    # Build transcript
    interview_log = "\n".join([
        f"{'HR' if isinstance(m, AIMessage) else student_name}: {m.content[:400]}"
        for m in history[1:] if not isinstance(m, SystemMessage)
    ])

    # Evaluate
    print("\n  Evaluating your performance...\n")
    result = evaluate_outcome(llm, interview_log, student_name, career_goal, company_label)

    # Print outcome + coaching
    print_outcome(result, student_name, career_goal, company_label, company_type)

    # Generate PDF report
    print("\n  Generating your interview report...")
    report_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = generate_report(
        student_name  = student_name,
        career_goal   = career_goal,
        company_type  = company_type,
        company_label = company_label,
        scores        = result["scores"],
        outcome       = result["outcome"],
        reason        = result["reason"],
        feedback      = result["feedback"],
        interview_log = interview_log,
        output_dir    = report_dir
    )
    print(f"\n  ✅ Report saved: {report_path}\n")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                      EXCELA                              ║
║   Excellence & Competency for Learning,                  ║
║        Achievement & Talent                              ║
║   Petra Christian University — CEC                       ║
╚══════════════════════════════════════════════════════════╝
""")

    student_name = input("  Your name: ").strip()
    career_goal  = input("  Target role (e.g. Finance Staff, Software Engineer, Marketing): ").strip()

    print("""
  Target company type:
  [1] National      — Indonesian companies (BCA, Telkom, Astra, Mandiri, Gojek...)
  [2] International — Fortune 500 (Google, Unilever, McKinsey, P&G, EY...)
""")
    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice == "1":
            company_type = "national"
            break
        elif choice == "2":
            company_type = "international"
            break
        else:
            print("  Please enter 1 or 2")

    docs     = scrape_job_data(career_goal, company_type)
    vectordb = build_rag(docs)

    input("  Press Enter when ready to begin your interview...")
    run_interview(student_name, career_goal, company_type, vectordb)


if __name__ == "__main__":
    main()