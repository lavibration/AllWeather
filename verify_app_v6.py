import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 2000})
        try:
            await page.goto("http://localhost:8501", timeout=60000)
            await page.wait_for_selector("[data-testid='stMetricValue']", timeout=45000)
            await asyncio.sleep(5)
            await page.screenshot(path="v6_tab1_full.png", full_page=True)
            print("Captured Full Tab 1")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

asyncio.run(run())
