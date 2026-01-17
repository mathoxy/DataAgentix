from smolagents import tool
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet
import os

@tool
def generate_pdf_report(
    title: str,
    sections: list,
    images: list,
) -> str:
    """
    Generate a PDF report with text sections and images.
    Args:
        title (str): Title of the report.

        sections (list): List of dictionaries with "title" and "content" keys for each section.
        images (list): List of image file paths to include in the report.
    Returns:
        str: Path to the generated PDF report.
    """
    output_path: str = f"reports/{title.replace(' ', '_')}.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))

    # Sections
    for section in sections:
        story.append(Paragraph(section["title"], styles["Heading2"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(section["content"], styles["BodyText"]))
        story.append(Spacer(1, 20))

    # Images
    for img_path in images:
        if os.path.exists(img_path):
            story.append(Paragraph("Figure", styles["Heading3"]))
            story.append(Image(img_path, width=400, height=300))
            story.append(Spacer(1, 20))

    doc.build(story)
    return output_path
