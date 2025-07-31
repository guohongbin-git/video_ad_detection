
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import os
import time

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from . import config

# --- Register the Chinese font ---
FONT_NAME = "SourceHanSans"
FONT_PATH = os.path.join(config.MATERIALS_DIR, "SourceHanSansSC-Regular.otf")

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
else:
    print(f"WARNING: Font file not found at {FONT_PATH}. Chinese characters may not render correctly.")
    FONT_NAME = "Helvetica" # Fallback to default


def generate_report(report_data: dict):
    """
    Creates a detailed PDF report with side-by-side comparison screenshots and semantic descriptions.

    Args:
        report_data (dict): A dictionary containing all necessary data for the report.
    """
    recorded_video_path = report_data["recorded_video_path"]
    report_filename = f"report_{os.path.basename(recorded_video_path)}.pdf"
    report_path = os.path.join(config.REPORTS_DIR, report_filename)
    
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    styles['h1'].fontName = FONT_NAME
    styles['h2'].fontName = FONT_NAME
    styles['Normal'].fontName = FONT_NAME
    story.append(Paragraph("视频广告检测报告", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Summary Section ---
    story.append(Paragraph("<b>Summary of Findings</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    summary_text = (
        f"<b>- 被检测视频:</b> {os.path.basename(recorded_video_path)}<br/>"
        f"<b>- 最佳匹配素材:</b> {report_data['best_match_material_filename']}<br/>"
        f"<b>- 最高相似度分数:</b> {report_data['overall_similarity_score']:.2f}<br/>"
        f"<b>- 源素材时长:</b> {report_data['material_duration']:.2f} 秒<br/>"
        f"<b>- 视频中匹配总时长:</b> {report_data['total_matched_duration_in_recorded']:.2f} 秒"
    )
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Comparison Evidence Section ---
    story.append(Paragraph("<b>Comparative Evidence</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    screenshots = report_data.get("comparison_screenshots", [])
    if not screenshots:
        story.append(Paragraph("No comparison screenshots were generated.", styles['Normal']))
    else:
        # Create a table for side-by-side image comparison
        table_data = [["实拍视频帧", "原始素材帧"]]
        
        for item in screenshots:
            try:
                # Create Image objects
                img_recorded = Image(item["recorded_frame_path"], width=2.5*inch, height=1.5*inch, kind='proportional')
                img_material = Image(item["material_frame_path"], width=2.5*inch, height=1.5*inch, kind='proportional')
                
                # Create timestamp and description paragraphs
                ts_desc_recorded = Paragraph(f"时间: {item['recorded_time']:.2f}s<br/>描述: {item['recorded_frame_description']}", styles['Normal'])
                ts_desc_material = Paragraph(f"时间: {item['material_time']:.2f}s<br/>描述: {item['material_frame_description']}", styles['Normal'])
                
                # Add images and timestamps/descriptions to the table row
                table_data.append([[img_recorded, ts_desc_recorded], [img_material, ts_desc_material]])

            except Exception as img_e:
                print(f"Error processing image for PDF: {img_e}")
                # Add placeholder text if an image fails to load
                table_data.append([f"Error loading image: {os.path.basename(item['recorded_frame_path'])}", 
                                   f"Error loading image: {os.path.basename(item['material_frame_path'])}"])

        # Define table style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])

        # Create and add the table to the story
        comparison_table = Table(table_data, colWidths=[3.0 * inch, 3.0 * inch])
        comparison_table.setStyle(style)
        story.append(comparison_table)

    # --- Build the PDF ---
    try:
        doc.build(story)
        print(f"Successfully generated report at: {report_path}")
    except Exception as e:
        print(f"Error building PDF report: {e}")

    # --- Clean up screenshot files ---
    all_screenshots_to_delete = []
    for item in screenshots:
        all_screenshots_to_delete.append(item["recorded_frame_path"])
        all_screenshots_to_delete.append(item["material_frame_path"])
    
    for screenshot_path in set(all_screenshots_to_delete): # Use set to avoid deleting the same file twice
        if screenshot_path and os.path.exists(screenshot_path):
            try:
                os.remove(screenshot_path)
            except OSError as e:
                print(f"Error removing screenshot file {screenshot_path}: {e}")

    return report_filename

