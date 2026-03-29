import asyncio
from playwright.async_api import async_playwright
import os

async def capture_full_dashboard():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 1600})

        # Streamlit usually runs on 8501
        url = "http://localhost:8501"
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(5) # Wait for charts to render

            # Cockpit Tab (Default) - Scroll down to see Regime Graphs
            await page.screenshot(path="v4_cockpit_full.png")

            # Signals Tab
            await page.get_by_text("Signaux & Hystérésis").click()
            await asyncio.sleep(2)
            await page.screenshot(path="v4_signals.png")

            # Performance Tab
            await page.get_by_text("Analyse des Performances").click()
            await asyncio.sleep(2)
            await page.screenshot(path="v4_performance.png")

            # Methodology Tab
            await page.get_by_text("Méthodologie").click()
            await asyncio.sleep(1)
            await page.screenshot(path="v4_methodology.png")

            print("Screenshots captured successfully.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    if not os.path.exists("verification"):
        os.makedirs("verification")
    os.chdir("verification")
    asyncio.run(capture_full_dashboard())
