from playwright.sync_api import Page, expect, sync_playwright
import os

def verify_dashboard(page: Page):
    page.goto("http://localhost:8501")
    # Wait for the app to load
    page.wait_for_selector("text=Cockpit d'Exécution", timeout=20000)
    page.wait_for_timeout(5000) # Give more time for Plotly

    # Final screenshot for main verification
    page.screenshot(path="/home/jules/verification/verification_new_charts_full.png", full_page=True)
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 2400})
        page = context.new_page()
        try:
            verify_dashboard(page)
        finally:
            context.close()
            browser.close()
