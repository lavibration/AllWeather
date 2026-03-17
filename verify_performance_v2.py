import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 2000})
        await page.goto("http://localhost:8501")
        await page.wait_for_timeout(5000)
        await page.click("text=Performance")
        await page.wait_for_timeout(3000)
        await page.screenshot(path="/home/jules/verification/performance_v2.png")
        await browser.close()

asyncio.run(run())
