import asyncio
from playwright.async_api import async_playwright
import os

async def capture_full_dashboard():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 2000})

        url = "http://localhost:8501"
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(5)

            # Use get_by_role("tab") to avoid ambiguity
            tabs = ["Cockpit d'Exécution", "Signaux & Hystérésis", "Analyse des Performances", "Méthodologie"]

            for i, tab_name in enumerate(tabs):
                await page.get_by_role("tab", name=tab_name).click()
                await asyncio.sleep(3)
                # Scroll down to ensure all charts (especially Regime Graphs) are rendered and visible
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)
                await page.screenshot(path=f"v4_tab_{i}_{tab_name.replace(' ', '_')}.png", full_page=True)

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
