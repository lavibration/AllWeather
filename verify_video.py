from playwright.sync_api import sync_playwright
import os
import time

def run_cuj(page):
    page.goto("http://localhost:8501")
    page.wait_for_timeout(5000) # Wait for initial load

    # Tab 1: Cockpit
    page.evaluate("window.scrollTo(0, 500)")
    page.wait_for_timeout(1000)
    page.evaluate("window.scrollTo(0, 1000)")
    page.wait_for_timeout(1000)

    # Tab 2: Signals
    page.get_by_role("tab", name="Signaux & Hystérésis").click()
    page.wait_for_timeout(2000)
    page.evaluate("window.scrollTo(0, 500)")
    page.wait_for_timeout(1000)

    # Tab 3: Performance
    page.get_by_role("tab", name="Analyse des Performances").click()
    page.wait_for_timeout(2000)
    page.evaluate("window.scrollTo(0, 500)")
    page.wait_for_timeout(1000)

    # Tab 4: Methodology
    page.get_by_role("tab", name="Méthodologie").click()
    page.wait_for_timeout(2000)

    # Take final screenshot
    page.screenshot(path="/home/jules/verification/screenshots/final_verification.png")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    if not os.path.exists("/home/jules/verification/videos"):
        os.makedirs("/home/jules/verification/videos")
    if not os.path.exists("/home/jules/verification/screenshots"):
        os.makedirs("/home/jules/verification/screenshots")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos",
            viewport={'width': 1280, 'height': 1200}
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
