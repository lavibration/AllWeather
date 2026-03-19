from playwright.sync_api import sync_playwright, expect
import os

def verify_cb(page):
    page.goto("http://localhost:8501")
    page.wait_for_timeout(2000) # Wait for streamlit to load

    # Check for Circuit Breaker mention in Cockpit or Methodology
    # Tab 4: Methodology
    page.get_by_role("tab", name="📜 Méthodologie").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="/home/jules/verification/cb_methodology.png")

    # Tab 1: Cockpit (Check if current status is displayed)
    page.get_by_role("tab", name="🚀 Cockpit").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="/home/jules/verification/cb_cockpit.png")

    # Tab 3: Performance (Check new MDD)
    page.get_by_role("tab", name="📈 Performance").click()
    page.wait_for_timeout(1000)
    page.screenshot(path="/home/jules/verification/cb_performance.png")

if __name__ == "__main__":
    os.makedirs("/home/jules/verification/video", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(record_video_dir="/home/jules/verification/video")
        page = context.new_page()
        try:
            verify_cb(page)
        finally:
            context.close()
            browser.close()
