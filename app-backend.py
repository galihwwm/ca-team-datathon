from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from time import time

from code_28 import (
    parse_pdf_to_markdown_with_images,
    load_or_create_index,
    load_cem_workunits,
    load_developer_actions,
    EvidenceAgent,
    PartAgent,
    WorkUnitRetriever,
    DeveloperActionRetriever,
    NavigatorAgent,
    EvaluationAgent,
    DeveloperAgent,
    UserAgent,
    ReportGenerator
)

from imports_and_helpers3_patched import sanitize_filename

app = Flask(__name__)
CORS(app)
from flask import send_from_directory

# Serve file dari folder outputs
@app.route('/outputs/<path:filename>')
def download_file(filename):
    return send_from_directory('outputs', filename)


# Caching simulation
standard_indexes = {}
historical_index = None
workunit_database = None
developer_database = None

@app.before_request
def load_resources():
    global standard_indexes, historical_index, workunit_database, developer_database
    if standard_indexes and historical_index and workunit_database and developer_database:
        return  # Sudah dimuat

    for i, name in enumerate(["part1", "part2", "part3", "part4", "part5"], 1):
        std_path = f"d4dproject/standard/CC2022PART{i}R1.pdf"
        std_chunks, _ = parse_pdf_to_markdown_with_images(std_path)
        standard_indexes[name] = load_or_create_index(std_chunks, sanitize_filename(name))

    hist_chunks = []
    for fname in os.listdir("d4dproject/historicalST"):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join("d4dproject/historicalST", fname)
            chunks, _ = parse_pdf_to_markdown_with_images(full_path)
            hist_chunks.extend(chunks)
    historical_index = load_or_create_index(hist_chunks, sanitize_filename("historical_index"))

    workunit_database = load_cem_workunits("d4dproject/standard/CEM2022R1.pdf")
    developer_database = load_developer_actions("d4dproject/standard/CC2022PART3R1.pdf")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        token = request.form.get("token")
        mode = request.form.get("mode")
        input_text = request.form.get("input_text")
        ase_family = request.form.get("ase_family")
        file = request.files.get("file")

        if not token or not mode or not file:
            return jsonify({"error": "Missing required fields."}), 400

        role = "Developer" if token.lower().startswith("dev") else "Evaluator"

        os.makedirs("uploads", exist_ok=True)
        unique_filename = f"{uuid.uuid4().hex}_{sanitize_filename(file.filename)}"
        saved_path = os.path.join("uploads", unique_filename)
        file.save(saved_path)

        chunks, _ = parse_pdf_to_markdown_with_images(saved_path)
        evidence_index = load_or_create_index(chunks, sanitize_filename(f"evidence_{uuid.uuid4().hex[:6]}"))

        evidence_agent = EvidenceAgent(evidence_index)
        part_agent = PartAgent(standard_indexes)
        workunit_retriever = WorkUnitRetriever(workunit_database)
        developer_retriever = DeveloperActionRetriever(developer_database)
        navigator = NavigatorAgent(role)
        evaluation_agent = EvaluationAgent(part_agent, historical_index, evidence_index)
        developer_agent = DeveloperAgent(part_agent, historical_index, evidence_index)

        user_agent = UserAgent(
            role, evidence_agent, part_agent,
            workunit_retriever, developer_retriever,
            navigator, evaluation_agent, developer_agent
        )

        if mode.lower() == "chatbot":
            if not input_text:
                return jsonify({"error": "Input text is required for chatbot mode."}), 400
            result = user_agent.process_query(input_text)
            return jsonify({"role": role, "response": result})

        elif mode.lower() == "report":
            if not ase_family:
                return jsonify({"error": "ASE family is required for report generation."}), 400

            prefix = ase_family.split(".")[0]
            try:
                if role == "Evaluator":
                    workunits = workunit_retriever.retrieve_family(prefix)
                    results = evaluation_agent.evaluate_cem(workunits)
                 
                else:
                    dev_actions = developer_retriever.retrieve_family(prefix)
                    results = developer_agent.guide_development("", dev_actions)
                  

                report = ReportGenerator(prefix=f"{role.lower()}_report")
                report.add_results(results, result_type="Evaluation" if role == "Evaluator" else "Developer")
                report_path = report.save()

                return jsonify({
                    "role": role,
                    "ase_family": ase_family,
                    "results": results,
                    "report_path": report_path
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            return jsonify({"error": "Invalid mode specified."}), 400

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
