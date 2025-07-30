from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os
import cv2
import time # Import time module
from . import config

def create_report(recorded_video_path: str, matched_ad: str, similarity: float):
    """
    Creates a PDF report with the ad detection results using ReportLab.

    Args:
        recorded_video_path (str): Path to the recorded video.
        matched_ad (str): Filename of the matched ad.
        similarity (float): The similarity score.
    """
    report_filename = f"report_{os.path.basename(recorded_video_path)}.pdf"
    report_path = os.path.join(config.REPORTS_DIR, report_filename)
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Ad Detection Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Report Details
    story.append(Paragraph(f"<b>Recorded Video:</b> {os.path.basename(recorded_video_path)}", styles['Normal']))
    story.append(Paragraph(f"<b>Matched Ad:</b> {matched_ad}", styles['Normal']))
    story.append(Paragraph(f"<b>Similarity Score:</b> {similarity:.2f}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Add screenshots
    screenshots = take_screenshots(recorded_video_path)
    if screenshots:
        story.append(Paragraph("<b>Evidence Screenshots:</b>", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))
        for i, screenshot_path in enumerate(screenshots):
            try:
                img = Image(screenshot_path, width=3*inch, height=2*inch) # Adjust size as needed
                story.append(img)
                story.append(Spacer(1, 0.1 * inch))
            except Exception as img_e:
                print(f"Error adding image {screenshot_path} to PDF: {img_e}")

    doc.build(story)
    print(f"Report generated at: {report_path}")

    # Clean up screenshot files after PDF is built
    for screenshot_path in screenshots:
        try:
            os.remove(screenshot_path)
        except OSError as e:
            print(f"Error removing screenshot {screenshot_path}: {e}")

def take_screenshots(video_path: str):
    """
    Takes a few screenshots from the video.

    Args:
        video_path (str): The path to the video.

    Returns:
        list: A list of paths to the screenshot images.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    screenshot_paths = []
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)

    # Clean up the video filename for safe use in paths
    cleaned_video_filename = "".join(c for c in os.path.basename(video_path) if c.isalnum() or c in ('.', '_', '-')).replace(' ', '_')

    for i in range(config.NUM_SCREENSHOTS):
        frame_index = int(total_frames / (config.NUM_SCREENSHOTS + 1) * (i + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            screenshot_path = os.path.join(config.SCREENSHOTS_DIR, f"screenshot_{cleaned_video_filename}_{i}.jpg")
            cv2.imwrite(screenshot_path, frame)
            # Verify file existence and size
            if not os.path.exists(screenshot_path):
                print(f"Error: Screenshot file not found after writing: {screenshot_path}")
                continue
            if os.path.getsize(screenshot_path) == 0:
                print(f"Error: Screenshot file is empty after writing: {screenshot_path}")
                continue
            time.sleep(0.1) # Small delay to ensure file is fully written
            screenshot_paths.append(screenshot_path)

    cap.release()
    return screenshot_paths