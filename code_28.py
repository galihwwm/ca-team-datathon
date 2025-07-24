# --- Import Shared Helpers ---
from imports_and_helpers3_patched import *
import base64
import mimetypes
import textwrap
from PIL import Image

# --- Chunking Functions ---
def heading_chunk_pdf(filepath):
    doc = pymupdf.open(filepath)
    chunks = []
    current_chunk = ""
    current_heading = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"] if span["text"].strip())
                if not text:
                    continue
                font_size = line["spans"][0]["size"]
                is_heading = font_size > 12 or text.strip().isdigit() or text.strip().startswith(('1.', '2.', '3.'))
                if is_heading:
                    if current_chunk:
                        chunks.append(Document(text=current_chunk.strip(), metadata={"heading": current_heading}))
                        current_chunk = ""
                    current_heading = text
                current_chunk += text + " "
    if current_chunk:
        chunks.append(Document(text=current_chunk.strip(), metadata={"heading": current_heading}))
    return chunks

#--- Legacy CEM & Developer Action Loading ---
def load_cem_workunits(filepath):
    WORK_UNIT_PATTERN = r"(ASE|ADV|AGD|ALC|ATE|AVA)_[A-Z]+\.\d+-\d+"
    doc = pymupdf.open(filepath)
    workunit_database = {}
    current_workunit_code = None
    current_content = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"] if span["text"].strip())
                if not text:
                    continue
                match = re.search(WORK_UNIT_PATTERN, text)
                if match:
                    if current_workunit_code:
                        workunit_database[current_workunit_code] = current_content.strip()
                        current_content = ""
                    current_workunit_code = match.group(0)
                current_content += text + " "
    if current_workunit_code:
        workunit_database[current_workunit_code] = current_content.strip()
    return workunit_database

def load_developer_actions(filepath):
    DEV_ACTION_PATTERN = r"(ASE|ADV|AGD|ALC|ATE|AVA)_[A-Z]+\.\d+\.\d+[DC]"
    doc = pymupdf.open(filepath)
    dev_action_database = {}
    current_action_code = None
    current_content = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"] if span["text"].strip())
                if not text:
                    continue
                match = re.search(DEV_ACTION_PATTERN, text)
                if match:
                    if current_action_code:
                        dev_action_database[current_action_code] = current_content.strip()
                        current_content = ""
                    current_action_code = match.group(0)
                current_content += text + " "
    if current_action_code:
        dev_action_database[current_action_code] = current_content.strip()
    return dev_action_database


# --- Build or Load Index ---
def load_or_create_index(documents, index_name):
    db_path = f"./chroma_{index_name}_db"
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_or_create_collection(index_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not collection.count():
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    else:
        return VectorStoreIndex.from_vector_store(vector_store)

# --- Agents ---
class EvidenceAgent:
    def __init__(self, evidence_index):
        retriever = VectorIndexRetriever(index=evidence_index, similarity_top_k=3)
        self.query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=CompactAndRefine())

    def query(self, question):
        return self.query_engine.query(question)

class PartAgent:
    def __init__(self, standard_indexes):
        self.standard_query_engines = {}
        for name, index in standard_indexes.items():
            retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
            self.standard_query_engines[name] = RetrieverQueryEngine(retriever=retriever, response_synthesizer=CompactAndRefine())

    def query(self, question):
        results = {}
        for name, engine in self.standard_query_engines.items():
            results[name] = engine.query(question)
        return results

class WorkUnitRetriever:
    def __init__(self, workunit_database):
        self.workunit_database = workunit_database

    def retrieve_family(self, family_prefix):
        return {code: text for code, text in self.workunit_database.items() if code.startswith(family_prefix)}

class DeveloperActionRetriever:
    def __init__(self, developer_actions):
        self.developer_actions = developer_actions

    def retrieve_family(self, family_prefix):
        return {code: text for code, text in self.developer_actions.items() if code.startswith(family_prefix)}

class NavigatorAgent:
    def __init__(self, role):
        self.role = role

    def decide_target(self, question):
        if self.role == "Developer":
            if re.search(r"(ASE|ADV|AGD|ALC|ATE|AVA)_[A-Z]+\.\d", question):
                return "DeveloperAction"
            else:
                return "Parts"
        
        elif self.role == "Evaluator":
            if re.search(r"(ASE|ADV|AGD|ALC|ATE|AVA)_[A-Z]+\.\d", question):
                return "CEM"
            else:
                return "Parts"
        
        else:
            return "Parts"

    def detect_family_prefix(self, question):
        match = re.search(r"(ASE|ADV|AGD|ALC|ATE|AVA)_[A-Z]+", question)
        return match.group(0) if match else None

class EvaluationAgent:
    def __init__(self, part_agent, historical_index, evidence_index, react_agent=None):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.part_agent = part_agent
        self.historical_index = historical_index
        self.react_agent = react_agent
        self.query_engine = evidence_index.as_query_engine(similarity_top_k=6)
        self.MAX_CHARS = 15000

    def _truncate(self, text):
        return text if len(text) <= self.MAX_CHARS else text[:self.MAX_CHARS]

    def retrieve_evidence_chunks(self, workunit_code, workunit_text, top_k=5):
        query_text = self._truncate(f"{workunit_code}: {workunit_text}")
        response = self.query_engine.query(query_text)
        return [node.text.strip() for node in response.source_nodes[:top_k]]

    def retrieve_background(self, workunit_code):
        response_dict = self.part_agent.query(workunit_code)
        parts = [r.response for r in response_dict.values()]
        return "\n\n".join(parts)

    def retrieve_historical_examples(self, workunit_code):
        retriever = self.historical_index.as_query_engine(similarity_top_k=3)
        response = retriever.query(workunit_code)
        return response.response if hasattr(response, "response") else str(response)

    def llm_evaluate_workunit(self, workunit_code, workunit_text):
        # === Retrieve relevant evidence
        top_chunks = self.retrieve_evidence_chunks(workunit_code, workunit_text)
        evidence_excerpt = self._truncate("\n\n".join(top_chunks))

        # === Retrieve background knowledge and historical examples
        background_text = self._truncate(self.retrieve_background(workunit_code))
        historical_examples = self._truncate(self.retrieve_historical_examples(workunit_code))

        # === Construct prompt
        prompt = f"""
You are acting as a Common Criteria Evaluator and your task is to evaluate the Security Target (ST) based on the following work unit expectations derived from the Common Evaluation Methodology (CEM).

Please base your response **strictly on the provided evidence** and relate each justification clearly to the corresponding evidence text.

WORK UNIT ({workunit_code}):
{workunit_text}

📄 EVIDENCE EXCERPT (from the uploaded file):
{evidence_excerpt}

📚 BACKGROUND KNOWLEDGE (CEM or CC Part 3):
{background_text}

✅ GOOD EXAMPLES (Historical STs):
{historical_examples}

Respond in the following format:
- **Compliance Status**: [Compliant / Partially Compliant / Not Compliant / Not Applicable]
- **Justification**: Explain why the evidence satisfies or fails the CEM criteria, with references to specific phrases or sections in the evidence.
- **Observations**: Mention ambiguities, risks, or unclear areas.
- **Suggestions**: Describe corrective actions for improvement.

Be critical, accurate, and reference specific parts of the ST.
""".strip()

        # === Run LLM
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content, evidence_excerpt

    def evaluate_cem(self, workunit_dict):
        results = []
        for code, text in workunit_dict.items():
            evaluation, excerpt = self.llm_evaluate_workunit(code, text)
            results.append({
                "workunit": code,
                "evaluation": evaluation,
                "evidence_excerpt_used": excerpt
            })
        return results

class DeveloperAgent:
    def __init__(self, part_agent, historical_index, evidence_index, react_agent=None):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.part_agent = part_agent
        self.historical_index = historical_index
        self.react_agent = react_agent
        self.query_engine = evidence_index.as_query_engine(similarity_top_k=6)
        self.MAX_CHARS = 15000

    def _truncate(self, text):
        return text if len(text) <= self.MAX_CHARS else text[:self.MAX_CHARS]

    def retrieve_evidence_chunks(self, dev_action_code, dev_action_text, top_k=5):
        query_text = self._truncate(f"{dev_action_code}: {dev_action_text}")
        response = self.query_engine.query(query_text)
        return [node.text.strip() for node in response.source_nodes[:top_k]]

    def retrieve_background(self, dev_action_code):
        response_dict = self.part_agent.query(dev_action_code)
        parts = [r.response for r in response_dict.values()]
        return "\n\n".join(parts)

    def retrieve_historical_examples(self, dev_action_code):
        retriever = self.historical_index.as_query_engine(similarity_top_k=3)
        response = retriever.query(dev_action_code)
        return response.response if hasattr(response, "response") else str(response)

    def llm_developer_guidance(self, dev_action_code, dev_action_text):
        # === Retrieve relevant evidence
        top_chunks = self.retrieve_evidence_chunks(dev_action_code, dev_action_text)
        evidence_excerpt = self._truncate("\n\n".join(top_chunks))

        # === Retrieve background knowledge and historical examples
        background_text = self._truncate(self.retrieve_background(dev_action_code))
        historical_examples = self._truncate(self.retrieve_historical_examples(dev_action_code))

        # === Construct prompt
        prompt = f"""
You are assisting a developer improving their Security Target (ST) to meet Common Criteria certification standards.

Please provide recommendations **based on the provided evidence** and explain which specific parts of the evidence support your feedback.

DEVELOPER ACTION ({dev_action_code}):
{dev_action_text}

📄 EVIDENCE EXCERPT (from the uploaded file):
{evidence_excerpt}

📚 BACKGROUND KNOWLEDGE (CC Part 1–5):
{background_text}

✅ GOOD EXAMPLES (Historical STs):
{historical_examples}

Respond in the following format:
- **Specific Gaps or Strengths** (point to which paragraph or section they relate to)
- **Detailed Recommendations** (mention exactly what to add, revise, or strengthen)
- **Conclusion**, state clearly and explain why if sufficient or insufficient. Describe what’s missing using examples from good practices.

Be specific, do not generalize. Quote and compare where needed.
""".strip()

        # === Run LLM
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content, evidence_excerpt

    def guide_development(self, evidence_text, dev_action_dict):
        results = []
        for code, text in dev_action_dict.items():
            guidance, excerpt = self.llm_developer_guidance(code, text)
            results.append({
                "developer_action": code,
                "guidance": guidance,
                "evidence_excerpt_used": excerpt
            })
        return results

# Optional: Check if image is valid
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Validate file integrity
        return True
    except Exception as e:
        print(f"❌ PIL could not open image: {file_path} - {e}")
        return False

def analyze_image(image_path, caption, react_agent=None):
    mime_type, _ = mimetypes.guess_type(image_path)
    
    if not mime_type or not mime_type.startswith("image/"):
        print(f"⚠️ Skipping invalid image: {image_path} — not a valid image MIME type")
        return "[Skipped invalid image]"

    try:
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        prompt_text = f"""
You are assisting a Common Criteria evaluator. Analyze this image titled "{caption}".
Describe how it supports or clarifies compliance for the evaluation.

Focus on:
- What the image shows
- Describe the image in relation to the text
- Any limitations or missing info
Be clear and concise.
""".strip()

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Error analyzing image {image_path}: {e}")
        return f"[Error analyzing image: {e}]"

# --- Main User Agent ---
# --- Enhanced UserAgent with background support ---
class UserAgent:
    def __init__(self, role, evidence_agent, part_agent, workunit_retriever, developer_action_retriever, navigator_agent, evaluation_agent, developer_agent, general_agent=None):
        self.role = role
        self.evidence_agent = evidence_agent
        self.part_agent = part_agent
        self.workunit_retriever = workunit_retriever
        self.developer_action_retriever = developer_action_retriever
        self.navigator_agent = navigator_agent
        self.evaluation_agent = evaluation_agent
        self.developer_agent = developer_agent
        self.general_agent = general_agent

    def _generate_chat_response(self, query):
        # Get relevant info from evidence and standards
        evidence_context = self.evidence_agent.query(query).response
        standard_contexts = self.part_agent.query(query)
        standard_text = "\n".join(result.response for result in standard_contexts.values())
        combined_context = (evidence_context + "\n" + standard_text).strip()
        prompt = f"""
    You are a Common Criteria assistant. Answer the following query based on context extracted from Security Target evidence and CC standards.

    Query:
    {query}

    Context:
    {combined_context}

    Provide a helpful, standards-based answer. If the context is unclear or incomplete, give a general but relevant explanation.
    If the question is not relevant to Common Criteria, you can answer: 
    Sorry, I am a Common Criteria assistant. I can only answer questions related to that.
    """.strip()

        answer = self.evaluation_agent.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return {
            "response": answer.choices[0].message.content.strip()
        }

    def process_query(self, question):
        evidence_response = self.evidence_agent.query(question)
        evidence_text = evidence_response.response if hasattr(evidence_response, "response") else str(evidence_response)
        target = self.navigator_agent.decide_target(question)
        background_knowledge = self.part_agent.query(question)

        if target == "CEM":
            family_prefix = self.navigator_agent.detect_family_prefix(question)
            if family_prefix:
                workunits = self.workunit_retriever.retrieve_family(family_prefix)
                core_results = self.evaluation_agent.evaluate_cem(workunits)
                for res in core_results:
                    res["background"] = background_knowledge
                return core_results
            else:
                return [self._generate_chat_response(question)]

        elif target == "DeveloperAction":
            family_prefix = self.navigator_agent.detect_family_prefix(question)
            if family_prefix:
                developer_actions = self.developer_action_retriever.retrieve_family(family_prefix)
                core_results = self.developer_agent.guide_development(evidence_text, developer_actions)
                for res in core_results:
                    res["background"] = background_knowledge
                return core_results
            else:
                return [self._generate_chat_response(question)]

        else:
            return [self._generate_chat_response(question)]

# --- Load and Prepare Everything ---
standard_files = {
    "part1": "d4dproject/standard/CC2022PART1R1.pdf",
    "part2": "d4dproject/standard/CC2022PART2R1.pdf",
    "part3": "d4dproject/standard/CC2022PART3R1.pdf",
    "part4": "d4dproject/standard/CC2022PART4R1.pdf",
    "part5": "d4dproject/standard/CC2022PART5R1.pdf",
}

cem_path = "d4dproject/standard/CEM2022R1.pdf"
developer_action_path = "d4dproject/standard/CC2022PART3R1.pdf"
evidence_path = "d4dproject/evidence/MyDigitalST.pdf"

print("\n⏳ Preparing standard indexes...")
standard_docs = []
standard_indexes = {}
for name, path in standard_files.items():
    docs, _ = parse_pdf_to_markdown_with_images(path)
    standard_docs.extend(docs)
    standard_indexes[name] = load_or_create_index(docs, name)

print("\n⏳ Preparing CEM work unit database...")
workunit_database = load_cem_workunits(cem_path)

print("\n⏳ Preparing Developer action database...")
developer_action_database = load_developer_actions(developer_action_path)

# print("\n⏳ Preparing Evidence index...")
markdown_chunks, image_documents = parse_pdf_to_markdown_with_images(evidence_path)

basename = os.path.basename(evidence_path).replace(".pdf", "").replace(" ", "_")
evidence_index = load_or_create_index(markdown_chunks, f"evidence_{basename}")

# Optional: assemble all markdown text for any summary or fallback usage
evidence_docs = markdown_chunks
ase_evidence_text = "\n".join(doc.text for doc in markdown_chunks)

# === Load Historical Evidence ===
print("\n⏳ Preparing Historical ST index...")
historical_chunks = []
historical_folder = "d4dproject/historicalST"
for fname in os.listdir(historical_folder):
    if fname.lower().endswith(".pdf"):
        full_path = os.path.join(historical_folder, fname)
        chunks, _ = parse_pdf_to_markdown_with_images(full_path)
        # 🔧 Strip large metadata fields to avoid LlamaIndex chunk error
        for doc in chunks:
            doc.metadata = {
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0)
            }
        historical_chunks.extend(chunks)
historical_index = load_or_create_index(historical_chunks, "historical_index")

# --- Connect Agents ---
evidence_agent = EvidenceAgent(evidence_index)
part_agent = PartAgent(standard_indexes)
workunit_retriever = WorkUnitRetriever(workunit_database)
developer_action_retriever = DeveloperActionRetriever(developer_action_database)
navigator_agent = NavigatorAgent(role=None)
evaluation_agent = EvaluationAgent(part_agent=part_agent, historical_index=historical_index, evidence_index=evidence_index)
developer_agent = DeveloperAgent(part_agent=part_agent, historical_index=historical_index, evidence_index=evidence_index)