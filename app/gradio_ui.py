import gradio as gr
from gradio_client import file
from agents.agents import orchestrator_agent
import shutil
import os
from utils.prompt_refinement import refine_prompt

def run_gradio():
    def run_agent(file, user_message):
        if file is not None:
            os.makedirs("data/raw", exist_ok=True)
            file_name = os.path.basename(file.name)
            file_path = f"data/raw/{file_name}"
            shutil.copy(file.name, file_path)
        else:
            file_name = None

        # Refine the prompt before sending to agent (handles empty message and file name)
        refined_prompt = refine_prompt(user_message, file_name)
        # Run OrchestratorAgent
        response = orchestrator_agent.run(refined_prompt)

        # Try to extract PDF path from response (if any)
        report_path = None
        import re
        def extract_pdf_path(text):
            # Prioritize reports/filename.pdf, fallback to any .pdf path
            match = re.search(r"reports/[A-Za-z0-9_\-]+\.pdf", str(text))
            if match:
                return match.group(0)
            match = re.search(r"[\w/\\.\-]+\.pdf", str(text))
            if match:
                return match.group(0)
            return None

        if isinstance(response, dict):
            # Search all values in the dict for a .pdf path
            for v in response.values():
                if ".pdf" in str(v):
                    report_path = extract_pdf_path(v)
                    if report_path:
                        break
        elif ".pdf" in str(response):
            report_path = extract_pdf_path(response)
        print("Generated report path:", report_path)

        return response, report_path if report_path and os.path.exists(report_path) else None

    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("## DataAgentix")
        file_input = gr.File(label="Import your CSV file")
        text_input = gr.Textbox(label="Your request")
        output = gr.Textbox(label="DataAgent's response", lines=15)
        pdf_download = gr.File(label="Download PDF Report", interactive=False, visible=False)

        def update_download(response, report_path):
            # Show download if PDF exists
            if report_path:
                return gr.update(visible=True, value=report_path)
            else:
                return gr.update(visible=False, value=None)

        run_btn = gr.Button("Run")
        run_btn.click(
            run_agent,
            inputs=[file_input, text_input],
            outputs=[output, pdf_download]
        )
        # Update download visibility after run
        run_btn.click(
            update_download,
            inputs=[output, pdf_download],
            outputs=pdf_download
        )

    demo.launch()