import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto("http://localhost:8501", timeout=60000)

            # Wait for any metric to appear (indicates data loaded)
            await page.wait_for_selector("[data-testid='stMetricValue']", timeout=45000)
            await asyncio.sleep(5) # Extra buffer for Plotly

            # Tab 1
            await page.screenshot(path="v5_tab1.png")
            print("Captured Tab 1")

            # Tab 2
            await page.get_by_role("tab", name="Signaux & Hystérésis").click()
            await page.wait_for_selector("text=Signal Croissance US", timeout=30000)
            await asyncio.sleep(5)
            await page.screenshot(path="v5_tab2.png")
            print("Captured Tab 2")

            # Tab 3
            await page.get_by_role("tab", name="Analyse des Performances").click()
            await page.wait_for_selector("text=Statistiques Globales", timeout=30000)
            await asyncio.sleep(5)
            await page.screenshot(path="v5_tab3.png")
            print("Captured Tab 3")

            # Tab 4
            await page.get_by_role("tab", name="Méthodologie").click()
            await page.wait_for_selector("text=Structure du Portefeuille", timeout=30000)
            await page.screenshot(path="v5_tab4.png")
            print("Captured Tab 4")

        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path="v5_error.png")
        finally:
            await browser.close()

asyncio.run(run())
